"""
Multi-person video handling using SAM (Segment Anything Model).

When videos contain multiple people, this module isolates the center person
using SAM segmentation to enable accurate motion feature extraction.
"""

import os
import tempfile
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import logging

from .utils import get_video_info
from .yolo_person_detector import YOLOPersonDetector

logger = logging.getLogger(__name__)


class MultiPersonHandler:
    """
    Handles videos with multiple people by segmenting the center person.
    
    Uses SAM (Segment Anything Model) to create per-frame masks isolating
    the person closest to the horizontal center of the frame, with tracking
    to maintain consistency across frames.
    """
    
    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        model_type: str = "vit_h",
        device: str = "cuda",
        yolo_detector: Optional[YOLOPersonDetector] = None
    ):
        """
        Args:
            sam_checkpoint: Path to SAM model checkpoint
            model_type: SAM model type ("vit_h", "vit_l", or "vit_b")
            device: Device to run SAM on ("cuda" or "cpu")
            yolo_detector: YOLO person detector instance (will create if None)
        """
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device = device
        
        self._sam = None
        self._predictor = None
        self._download_failed = False
        self._load_attempted = False
        
        # YOLO detector for finding person bboxes
        self.yolo_detector = yolo_detector
        if self.yolo_detector is None:
            self.yolo_detector = YOLOPersonDetector()
    
    def cleanup(self):
        """Release GPU memory and clean up SAM model resources."""
        import torch
        import gc
        try:
            if self._predictor is not None:
                del self._predictor
                self._predictor = None
            if self._sam is not None:
                del self._sam
                self._sam = None
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            pass
    
    def _load_sam(self):
        """Lazy load SAM model."""
        if self._sam is not None:
            return
        
        # If download already failed, don't try again
        if self._download_failed:
            raise FileNotFoundError(
                f"SAM checkpoint download previously failed. "
                f"Please download manually and place at: ~/.cache/sam/sam_{self.model_type}.pth"
            )
        
        # If load was already attempted and failed, don't retry
        if self._load_attempted:
            return
        
        self._load_attempted = True
        logger.info(f"Loading SAM model ({self.model_type})...")
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            # Try to find checkpoint if not provided
            if self.sam_checkpoint is None:
                # Common checkpoint locations
                possible_paths = [
                    f"checkpoints/sam_{self.model_type}.pth",
                    f"~/.cache/sam/sam_{self.model_type}.pth",
                    f"/tmp/sam_{self.model_type}.pth",
                ]
                
                # Also check for hash-suffixed files
                hash_suffixes = {
                    "vit_h": "_4b8939",
                    "vit_l": "_0b3195",
                    "vit_b": "_01ec64",
                }
                hash_suffix = hash_suffixes.get(self.model_type, "")
                if hash_suffix:
                    possible_paths.extend([
                        f"checkpoints/sam_{self.model_type}{hash_suffix}.pth",
                        f"~/.cache/sam/sam_{self.model_type}{hash_suffix}.pth",
                    ])
                
                for path in possible_paths:
                    expanded_path = os.path.expanduser(path)
                    if os.path.exists(expanded_path):
                        self.sam_checkpoint = expanded_path
                        break
                
                # If not found, try to download it
                if self.sam_checkpoint is None:
                    logger.info(f"SAM checkpoint not found, attempting to download...")
                    cache_dir = os.path.expanduser("~/.cache/sam")
                    os.makedirs(cache_dir, exist_ok=True)
                    checkpoint_path = os.path.join(cache_dir, f"sam_{self.model_type}.pth")
                    
                    # SAM checkpoint URLs
                    checkpoint_urls = {
                        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    }
                    
                    if self.model_type not in checkpoint_urls:
                        raise ValueError(f"Unknown SAM model type: {self.model_type}")
                    
                    url = checkpoint_urls[self.model_type]
                    logger.info(f"Downloading SAM checkpoint from {url}...")
                    logger.info(f"This may take several minutes (~2.4GB for vit_h)...")
                    
                    try:
                        import requests
                        from tqdm import tqdm
                        
                        response = requests.get(url, stream=True, timeout=30)
                        response.raise_for_status()
                        
                        total_size = int(response.headers.get('content-length', 0))
                        
                        with open(checkpoint_path, 'wb') as f, tqdm(
                            desc=f"Downloading {os.path.basename(checkpoint_path)}",
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    bar.update(len(chunk))
                        
                        logger.info(f"Downloaded SAM checkpoint to {checkpoint_path}")
                        self.sam_checkpoint = checkpoint_path
                    except Exception as e:
                        logger.error(f"Failed to download SAM checkpoint: {e}")
                        self._download_failed = True
                        raise FileNotFoundError(
                            f"SAM checkpoint not found and download failed. "
                            f"Please download manually from: {url} "
                            f"and place it at: {checkpoint_path}"
                        )
            
            # Verify checkpoint exists before loading
            if not os.path.exists(self.sam_checkpoint):
                raise FileNotFoundError(
                    f"SAM checkpoint not found: {self.sam_checkpoint}. "
                    f"Please download it manually."
                )
            
            self._sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            self._sam.to(device=self.device)
            self._predictor = SamPredictor(self._sam)
            
            logger.info("SAM model loaded successfully")
            
        except (ImportError, FileNotFoundError) as e:
            self._load_attempted = False
            raise
        except Exception as e:
            self._load_attempted = False
            logger.error(f"Failed to load SAM model: {e}")
            raise
    
    def _find_center_person_bbox_yolo(
        self,
        frame: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find bounding box of the person closest to horizontal center using YOLO.
        
        Args:
            frame: BGR image array
            
        Returns:
            (x1, y1, x2, y2) bounding box or None if no person found
        """
        H, W = frame.shape[:2]
        center_x = W // 2
        
        # Run YOLO detection on single frame
        try:
            # Ensure model is loaded
            self.yolo_detector._load_model()
            
            # Run detection (YOLO expects RGB, but we have BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo_detector._model(frame_rgb, classes=[0], conf=0.25, verbose=False)
            
            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                return None
            
            # Get all person detections
            boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4) format: x1, y1, x2, y2
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Find person closest to center
            best_box = None
            min_dist_to_center = float('inf')
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                box_center_x = (x1 + x2) / 2
                dist_to_center = abs(box_center_x - center_x)
                
                if dist_to_center < min_dist_to_center:
                    min_dist_to_center = dist_to_center
                    best_box = (int(x1), int(y1), int(x2), int(y2))
            
            return best_box
            
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
            return None
    
    def _find_best_matching_person(
        self,
        frame: np.ndarray,
        tracked_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the best matching person bbox, prioritizing consistency with tracked bbox.
        
        If tracked_bbox exists, prefer the detection with highest IoU.
        Otherwise, pick the person closest to horizontal center.
        """
        H, W = frame.shape[:2]
        center_x = W // 2
        
        # Get all person detections using YOLO
        try:
            self.yolo_detector._load_model()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo_detector._model(frame_rgb, classes=[0], conf=0.25, verbose=False)
            
            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                return None
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Collect bboxes for each detected person
            person_bboxes = []
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                bbox_center_x = (x1 + x2) / 2
                person_bboxes.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center_x': bbox_center_x,
                    'dist_to_center': abs(bbox_center_x - center_x),
                    'confidence': conf
                })
            
            if not person_bboxes:
                return None
            
            # If we have a tracked bbox, prefer person with highest IoU
            if tracked_bbox is not None:
                best_match = None
                best_iou = 0.3  # Minimum IoU threshold for matching
                
                for person in person_bboxes:
                    iou = self._compute_iou(person['bbox'], tracked_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = person['bbox']
                
                if best_match is not None:
                    return best_match
            
            # Fall back to person closest to center
            person_bboxes.sort(key=lambda x: x['dist_to_center'])
            return person_bboxes[0]['bbox']
            
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
            return None
    
    def _compute_iou(self, bbox1, bbox2) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def segment_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Segment person using SAM with bounding box prompt.
        
        Args:
            frame: BGR image array
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            Binary mask (H, W) with person region as True
        """
        if self._sam is None:
            raise RuntimeError("SAM model not loaded. Call _load_sam() first.")
        
        # Convert BGR to RGB for SAM
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set image
        self._predictor.set_image(frame_rgb)
        
        # Predict with box prompt
        x1, y1, x2, y2 = bbox
        input_box = np.array([x1, y1, x2, y2])
        
        masks, scores, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True
        )
        
        # Return best mask
        best_idx = np.argmax(scores)
        return masks[best_idx]
    
    def apply_mask_to_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Apply mask to frame, replacing non-person areas with background.
        
        Args:
            frame: BGR image array
            mask: Binary mask (H, W)
            background_color: BGR color for background
            
        Returns:
            Masked frame
        """
        result = np.full_like(frame, background_color)
        result[mask] = frame[mask]
        return result
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        target_fps: Optional[float] = None
    ) -> str:
        """
        Process video to isolate center person using SAM with consistent tracking.
        
        Uses IoU-based tracking to ensure the same person is followed throughout
        the video, even if they move away from center temporarily.
        
        Args:
            video_path: Input video path
            output_path: Output video path (None = auto temp file)
            target_fps: Target FPS (None = preserve original)
            
        Returns:
            Path to processed video
        """
        logger.info(f"Processing multi-person video: {video_path}")
        
        # Try to load SAM once at the start
        try:
            self._load_sam()
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            logger.error("Skipping segmentation for this video. Please ensure SAM checkpoint is available.")
            return video_path
        
        if output_path is None:
            ext = Path(video_path).suffix
            fd, output_path = tempfile.mkstemp(suffix=ext, prefix="segmented_")
            os.close(fd)
        
        info = get_video_info(video_path)
        fps = target_fps if target_fps else info['fps']
        
        # Open video for reading
        cap = cv2.VideoCapture(video_path)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, fourcc, fps,
            (info['width'], info['height'])
        )
        
        frame_count = 0
        tracked_bbox = None  # The bbox of the person we're tracking
        last_mask = None  # Cache last mask for frames between detections
        
        logger.info("Starting segmentation with consistent person tracking...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update tracking every 5 frames for efficiency while maintaining consistency
            if frame_count % 5 == 0:
                new_bbox = self._find_best_matching_person(frame, tracked_bbox)
                if new_bbox is not None:
                    tracked_bbox = new_bbox
            
            if tracked_bbox is not None:
                try:
                    # Segment person
                    mask = self.segment_person(frame, tracked_bbox)
                    last_mask = mask
                    # Apply mask
                    frame = self.apply_mask_to_frame(frame, mask)
                except Exception as e:
                    # Use last mask if segmentation fails
                    if last_mask is not None:
                        frame = self.apply_mask_to_frame(frame, last_mask)
                    else:
                        if frame_count == 0 or frame_count % 100 == 0:
                            logger.warning(f"Segmentation failed at frame {frame_count}: {e}")
                        if "SAM checkpoint" in str(e) or "download" in str(e).lower() or "FileNotFoundError" in str(type(e)):
                            logger.error(f"SAM not available. Skipping segmentation for this video.")
                            break
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.debug(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        
        logger.info(f"Saved segmented video to: {output_path} ({frame_count} frames)")
        return output_path
