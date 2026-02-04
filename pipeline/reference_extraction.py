"""Stage 5: Reference image extraction."""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

from .utils import read_video_frames

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector using InsightFace (same as WAN-Animate)."""
    
    def __init__(self):
        self.app_640 = None
        self.app_320 = None
        self.app_160 = None
        self._initialized = False
    
    def _initialize(self):
        """Initialize InsightFace detectors with multiple resolutions."""
        if self._initialized:
            return
        
        try:
            from insightface.app import FaceAnalysis
            import os
            
            # Try to find InsightFace models (check common locations)
            # Note: InsightFace expects root to be the parent of 'models' directory
            # So if models are at: /path/to/insightface/models/antelopev2/
            # Then root should be: /path/to/insightface
            insightface_root_paths = [
                'models/ByteDance/InfiniteYou/supports/insightface',
                'models/InfiniteYou/insightface',
                os.path.expanduser('~/.insightface'),
            ]
            
            insightface_root = None
            for path in insightface_root_paths:
                # Resolve to absolute path for checking
                abs_path = os.path.abspath(path)
                # Check if the models directory exists under this root
                model_check_path = os.path.join(abs_path, 'models', 'antelopev2')
                if os.path.exists(model_check_path):
                    insightface_root = path  # Keep relative path for InsightFace
                    logger.info(f"Found InsightFace models at: {model_check_path}")
                    break
            
            if insightface_root is None:
                # Try default location (relative to current working directory)
                insightface_root = 'models/ByteDance/InfiniteYou/supports/insightface'
                logger.warning(f"InsightFace models not found in common locations, trying: {insightface_root}")
            
            # Initialize detectors with different resolutions (same as WAN-Animate)
            try:
                # Check if models exist - wait for download AND extraction to complete
                import time
                # Resolve to absolute path for file checking
                abs_root = os.path.abspath(insightface_root)
                model_dir = os.path.join(abs_root, 'models', 'antelopev2')
                
                # Check for nested directory (common extraction issue - zip contains antelopev2 folder)
                nested_dir = os.path.join(model_dir, 'antelopev2')
                if os.path.exists(nested_dir):
                    # Check if files are in nested directory
                    test_file = os.path.join(nested_dir, 'scrfd_10g_bnkps.onnx')
                    if os.path.exists(test_file):
                        logger.info(f"Found models in nested directory, moving to expected location...")
                        # Move files from nested directory to expected location
                        import shutil
                        try:
                            for file in os.listdir(nested_dir):
                                src = os.path.join(nested_dir, file)
                                dst = os.path.join(model_dir, file)
                                if os.path.isfile(src) and not os.path.exists(dst):
                                    shutil.move(src, dst)
                                    logger.debug(f"Moved {file} to expected location")
                            # Remove empty nested directory
                            try:
                                os.rmdir(nested_dir)
                            except:
                                pass
                            logger.info("Successfully moved models to expected location")
                        except Exception as move_error:
                            logger.warning(f"Failed to move models: {move_error}. Using nested directory.")
                            model_dir = nested_dir
                
                # Required model files that must exist
                required_files = [
                    'scrfd_10g_bnkps.onnx',  # Detection model (most important)
                    '1k3d68.onnx',
                    '2d106det.onnx',
                    'genderage.onnx',
                    'glintr100.onnx'
                ]
                
                # Wait up to 30 seconds for models to be downloaded and extracted
                models_ready = False
                for attempt in range(30):
                    # Check both normal and nested paths
                    check_dirs = [
                        os.path.join(abs_root, 'models', 'antelopev2'),
                        os.path.join(abs_root, 'models', 'antelopev2', 'antelopev2')
                    ]
                    
                    for check_dir in check_dirs:
                        if os.path.exists(check_dir):
                            # Check if required files exist
                            files_exist = all(
                                os.path.exists(os.path.join(check_dir, f)) 
                                for f in required_files
                            )
                            if files_exist:
                                model_dir = check_dir
                                models_ready = True
                                logger.debug(f"InsightFace models ready at {model_dir} after {attempt + 1} seconds")
                                break
                    
                    if models_ready:
                        break
                    
                    # Log which files are missing
                    if attempt % 5 == 0:  # Log every 5 seconds
                        missing = []
                        for check_dir in check_dirs:
                            if os.path.exists(check_dir):
                                missing.extend([f for f in required_files 
                                              if not os.path.exists(os.path.join(check_dir, f))])
                        if missing:
                            logger.debug(f"Waiting for InsightFace model extraction... Missing: {missing}")
                    time.sleep(1)
                
                if not models_ready:
                    missing_files = [f for f in required_files 
                                   if not os.path.exists(os.path.join(model_dir, f))]
                    raise FileNotFoundError(
                        f"InsightFace models not fully extracted at {model_dir}. "
                        f"Missing files: {missing_files}. "
                        f"Please manually extract antelopev2.zip to {model_dir}"
                    )
                
                # Initialize with providers - try CPU first if CUDA fails
                providers = ['CPUExecutionProvider']
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                
                logger.info(f"Initializing InsightFace with root={insightface_root}, models at {model_dir}")
                # Use relative path for root (same as WAN-Animate)
                self.app_640 = FaceAnalysis(
                    name='antelopev2',
                    root=insightface_root,  # Relative path, InsightFace will resolve it
                    providers=providers
                )
                self.app_640.prepare(ctx_id=0, det_size=(640, 640))
                
                self.app_320 = FaceAnalysis(
                    name='antelopev2',
                    root=insightface_root,
                    providers=providers
                )
                self.app_320.prepare(ctx_id=0, det_size=(320, 320))
                
                self.app_160 = FaceAnalysis(
                    name='antelopev2',
                    root=insightface_root,
                    providers=providers
                )
                self.app_160.prepare(ctx_id=0, det_size=(160, 160))
                
                self._initialized = True
                logger.info(f"Using InsightFace for face detection (models from: {insightface_root})")
            except Exception as e:
                logger.warning(f"Failed to initialize InsightFace: {e}")
                logger.debug(f"InsightFace error details: {type(e).__name__}: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                
                # Check if zip file exists but wasn't extracted
                abs_root = os.path.abspath(insightface_root)
                zip_path = os.path.join(abs_root, 'models', 'antelopev2.zip')
                model_dir = os.path.join(abs_root, 'models', 'antelopev2')
                if os.path.exists(zip_path):
                    logger.warning(
                        f"Found antelopev2.zip at {zip_path} but models weren't extracted. "
                        f"Please manually extract it: unzip {zip_path} -d {model_dir}"
                    )
                
                logger.info("Falling back to MediaPipe/OpenCV")
                self._initialize_fallback()
        except ImportError:
            logger.warning("InsightFace not available. Install with: pip install insightface")
            logger.info("Falling back to MediaPipe/OpenCV")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Fallback to MediaPipe or OpenCV DNN."""
        self._detector = None
        self._use_mediapipe = False
        
        # Try MediaPipe
        try:
            try:
                # Try newer import style
                from mediapipe.python.solutions import face_detection as mp_face_detection
            except (ImportError, AttributeError):
                # Try older import style
                import mediapipe as mp
                mp_face_detection = mp.solutions.face_detection
            
            self._detector = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.3
            )
            self._use_mediapipe = True
            logger.info("Using MediaPipe for face detection (fallback)")
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"MediaPipe not available: {e}")
            # Try OpenCV DNN
            try:
                prototxt_path = Path(__file__).parent / "face_detector" / "deploy.prototxt"
                model_path = Path(__file__).parent / "face_detector" / "res10_300x300_ssd_iter_140000.caffemodel"
                if prototxt_path.exists() and model_path.exists():
                    self._detector = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
                    logger.info("Using OpenCV DNN for face detection (fallback)")
                else:
                    logger.warning("No face detector available - will use first frame as fallback")
                    self._detector = None
            except Exception as e2:
                logger.warning(f"Failed to load fallback detector: {e2}")
                self._detector = None
        self._initialized = True
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, Optional[dict]]]:
        """
        Detect faces in a frame using InsightFace (same method as WAN-Animate).
        
        Args:
            frame: BGR image array
            
        Returns:
            List of (x, y, width, height, face_info) tuples where face_info contains landmarks and other info
        """
        self._initialize()
        
        faces = []
        
        # Use InsightFace if available (same as WAN-Animate)
        if self.app_640 is not None:
            # Try different resolutions (same as WAN-Animate _detect_face method)
            face_info = self.app_640.get(frame)
            if len(face_info) == 0:
                face_info = self.app_320.get(frame)
            if len(face_info) == 0:
                face_info = self.app_160.get(frame)
            
            if face_info:
                h, w = frame.shape[:2]
                for info in face_info:
                    bbox = info['bbox']  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    # Ensure valid coordinates
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 10 and height > 10:  # Minimum face size
                        # Store face info (landmarks, etc.) for orientation detection
                        faces.append((x1, y1, width, height, info))
        elif hasattr(self, '_use_mediapipe') and self._use_mediapipe and hasattr(self, '_detector') and self._detector is not None:
            # Fallback to MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                results = self._detector.process(rgb_frame)
                if results and hasattr(results, 'detections') and results.detections:
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = max(0, int(bbox.xmin * w))
                        y = max(0, int(bbox.ymin * h))
                        width = min(w - x, int(bbox.width * w))
                        height = min(h - y, int(bbox.height * h))
                        if width > 10 and height > 10:
                            faces.append((x, y, width, height, None))  # No landmarks for MediaPipe
            except Exception as e:
                logger.debug(f"MediaPipe detection error: {e}")
        elif hasattr(self, '_detector') and self._detector is not None:
            # Fallback to OpenCV DNN
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            self._detector.setInput(blob)
            detections = self._detector.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.3:
                    x1 = max(0, int(detections[0, 0, i, 3] * w))
                    y1 = max(0, int(detections[0, 0, i, 4] * h))
                    x2 = min(w, int(detections[0, 0, i, 5] * w))
                    y2 = min(h, int(detections[0, 0, i, 6] * h))
                    width = x2 - x1
                    height = y2 - y1
                    if width > 0 and height > 0:
                        faces.append((x1, y1, width, height))
        
        return faces
    
    def evaluate_face_orientation(self, face_info: Optional[dict], frame_shape: Tuple[int, int], face_bbox: Tuple[int, int, int, int]) -> float:
        """
        Evaluate if face is front-facing using landmarks.
        
        Args:
            face_info: InsightFace face info dict with landmarks (or None)
            frame_shape: (height, width) of frame
            face_bbox: (x, y, width, height) bounding box
            
        Returns:
            Orientation score [0, 1] where 1.0 = perfectly front-facing, 0.0 = profile/extreme angle
        """
        if face_info is None or 'kps' not in face_info:
            # No landmarks available, assume neutral score
            return 0.5
        
        try:
            # Get 5 facial landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            kps = face_info['kps']  # Shape: (5, 2) - [x, y] coordinates
            
            if len(kps) < 5:
                return 0.5
            
            # Calculate face orientation indicators
            h, w = frame_shape
            
            # 1. Eye symmetry: distance from face center to each eye should be similar
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]
            
            face_center_x = face_bbox[0] + face_bbox[2] / 2
            face_center_y = face_bbox[1] + face_bbox[3] / 2
            
            # Distance from face center to each eye
            left_eye_dist = np.sqrt((left_eye[0] - face_center_x)**2 + (left_eye[0] - face_center_y)**2)
            right_eye_dist = np.sqrt((right_eye[0] - face_center_x)**2 + (right_eye[0] - face_center_y)**2)
            
            # Symmetry score: eyes should be equidistant from center
            eye_symmetry = 1.0 - min(1.0, abs(left_eye_dist - right_eye_dist) / max(left_eye_dist, right_eye_dist, 1.0))
            
            # 2. Nose position: should be near face center horizontally
            nose_center_dist = abs(nose[0] - face_center_x) / (face_bbox[2] / 2) if face_bbox[2] > 0 else 1.0
            nose_center_score = max(0, 1.0 - nose_center_dist)
            
            # 3. Eye alignment: eyes should be roughly horizontal (similar y-coordinates)
            eye_y_diff = abs(left_eye[1] - right_eye[1])
            eye_y_score = max(0, 1.0 - eye_y_diff / max(abs(left_eye[1] - face_center_y), 1.0))
            
            # Combined orientation score (front-facing = high score)
            orientation_score = (eye_symmetry * 0.4 + nose_center_score * 0.3 + eye_y_score * 0.3)
            
            return max(0.0, min(1.0, orientation_score))
            
        except Exception as e:
            logger.debug(f"Error evaluating face orientation: {e}")
            return 0.5
    
    def evaluate_face_clarity(self, frame: np.ndarray, face_region: Tuple[int, int, int, int], face_info: Optional[dict] = None) -> float:
        """
        Evaluate face clarity using the same method as temp.py.
        
        Scores based on:
        - Face size (30 points): 5-15% of image area is ideal
        - Face position (20 points): centered faces are better
        - Face sharpness (30 points): Laplacian variance (100-500 is good)
        - Face contrast (20 points): std deviation (20-60 is good)
        
        Args:
            frame: BGR image array
            face_region: (x, y, width, height) bounding box
            
        Returns:
            Clarity score in [0, 1] (normalized from 0-100)
        """
        h, w = frame.shape[:2]
        x, y, fw, fh = face_region
        
        # Extract face region (grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        face_roi = gray[y:y+fh, x:x+fw]
        
        if face_roi.size == 0:
            return 0.0
        
        scores = []
        
        # 1. Face size score (30 points) - 5-15% of image area is ideal
        face_area = fw * fh
        image_area = w * h
        face_ratio = face_area / image_area if image_area > 0 else 0
        if 0.05 <= face_ratio <= 0.15:
            size_score = 1.0
        elif face_ratio < 0.05:
            size_score = face_ratio / 0.05  # Too small
        else:
            size_score = max(0, 1.0 - (face_ratio - 0.15) / 0.15)  # Too large
        scores.append(size_score * 30)
        
        # 2. Face position score (20 points) - centered faces are better
        face_center_x = x + fw / 2
        face_center_y = y + fh / 2
        center_dist_x = abs(face_center_x - w/2) / (w/2) if w > 0 else 1.0
        center_dist_y = abs(face_center_y - h/2) / (h/2) if h > 0 else 1.0
        ideal_y_ratio = 0.25  # Upper 25% of image
        y_dist = abs(face_center_y / h - ideal_y_ratio) if h > 0 else 1.0
        position_score = max(0, 1.0 - (center_dist_x * 0.5 + y_dist * 0.5))
        scores.append(position_score * 20)
        
        # 3. Face sharpness/blur detection (30 points) - Laplacian variance
        laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        if 100 <= laplacian_var <= 500:
            sharpness_score = 1.0
        elif laplacian_var < 100:
            sharpness_score = laplacian_var / 100  # Too blurry
        else:
            sharpness_score = max(0, 1.0 - (laplacian_var - 500) / 500)  # Over-sharpened/noisy
        scores.append(sharpness_score * 30)
        
        # 4. Face region contrast (20 points) - good contrast = clearer features
        face_contrast = np.std(face_roi.astype(float))
        if 20 <= face_contrast <= 60:
            contrast_score = 1.0
        elif face_contrast < 20:
            contrast_score = face_contrast / 20  # Too low
        else:
            contrast_score = max(0, 1.0 - (face_contrast - 60) / 60)  # Too high
        scores.append(contrast_score * 20)
        
        # Calculate final score (0-100) and normalize to [0, 1]
        total_score = sum(scores) if scores else 0
        total_score = max(0, min(100, total_score))
        
        return total_score / 100.0  # Normalize to [0, 1]
    
    def score_face_quality(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int, Optional[dict]]]) -> Tuple[float, Tuple[int, int, int, int, Optional[dict]]]:
        """
        Score face quality based on clarity evaluation and center position.
        
        Selects the face with the best combined score (clarity + center position + front-facing).
        Prioritizes center person when multiple faces are detected.
        
        Args:
            frame: BGR image array
            faces: List of (x, y, width, height, face_info) tuples
            
        Returns:
            Tuple of (best_score, best_face) where best_face is (x, y, width, height, face_info)
        """
        if not faces:
            return (0.0, None)
        
        h, w = frame.shape[:2]
        frame_center_x = w / 2
        frame_center_y = h / 2
        
        best_score = -1.0
        best_face = None
        
        for face_data in faces:
            if len(face_data) == 5:
                x, y, fw, fh, face_info = face_data
            else:
                # Handle old format (backward compatibility)
                x, y, fw, fh = face_data[:4]
                face_info = face_data[4] if len(face_data) > 4 else None
            
            # Clarity score
            clarity_score = self.evaluate_face_clarity(frame, (x, y, fw, fh), face_info)
            
            # Center position score (prioritize center person)
            face_center_x = x + fw / 2
            face_center_y = y + fh / 2
            center_dist_x = abs(face_center_x - frame_center_x) / (w / 2) if w > 0 else 1.0
            center_dist_y = abs(face_center_y - frame_center_y) / (h / 2) if h > 0 else 1.0
            center_score = max(0, 1.0 - (center_dist_x * 0.6 + center_dist_y * 0.4))  # Weight horizontal more
            
            # Combined score: clarity (70%) + center position (30%)
            combined_score = clarity_score * 0.7 + center_score * 0.3
            
            if combined_score > best_score:
                best_score = combined_score
                best_face = (x, y, fw, fh, face_info)
        
        return (best_score, best_face)
    
    def cleanup(self):
        """Clean up resources."""
        # InsightFace doesn't need explicit cleanup
        if hasattr(self, '_use_mediapipe') and self._use_mediapipe and hasattr(self, '_detector') and self._detector is not None:
            self._detector.close()
        if hasattr(self, '_detector'):
            self._detector = None


def extract_first_frame(clip_path: Path, output_path: Path) -> bool:
    """
    Extract first frame from video clip.
    
    Args:
        clip_path: Path to video clip
        output_path: Path to save extracted frame
        
    Returns:
        True if successful
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {clip_path}")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        logger.error(f"Cannot read first frame from: {clip_path}")
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), frame)
    
    if success:
        logger.debug(f"Extracted first frame: {output_path}")
    else:
        logger.error(f"Failed to save first frame: {output_path}")
    
    return success


def extract_best_face_frame(
    clip_path: Path,
    output_path: Path,
    face_detector: FaceDetector
) -> bool:
    """
    Extract frame with best face quality and segment face region.
    
    Args:
        clip_path: Path to video clip
        output_path: Path to save extracted face image
        face_detector: Face detector instance
        
    Returns:
        True if successful
    """
    best_frame = None
    best_score = -1.0
    best_faces = []
    frames_with_faces = 0
    total_frames = 0
    
    # Scan all frames for best face quality (using clarity evaluation)
    for frame_idx, frame in read_video_frames(str(clip_path)):
        total_frames += 1
        faces = face_detector.detect_faces(frame)
        
        if faces:
            frames_with_faces += 1
            # Score based on clarity + center position + front-facing orientation
            score, best_face_in_frame = face_detector.score_face_quality(frame, faces)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_faces = faces
                logger.debug(f"Frame {frame_idx}: New best face score = {score:.3f} (clarity + center + orientation)")
    
    # Log detection statistics
    logger.debug(f"Face detection stats for {clip_path.name}: {frames_with_faces}/{total_frames} frames had faces")
    if best_score > 0:
        logger.debug(f"Best face clarity score: {best_score:.3f} (0-1 scale, higher is clearer)")
    
    # If no faces detected, fallback to first frame (same as WAN-Animate behavior)
    if best_frame is None or not best_faces:
        logger.warning(f"No faces detected in: {clip_path} ({frames_with_faces}/{total_frames} frames had faces)")
        logger.info(f"Using first frame as fallback for: {clip_path.name}")
        return extract_first_frame(clip_path, output_path)
    
    # Select face with best combined score (clarity + center + front-facing)
    h, w = best_frame.shape[:2]
    
    # Use score_face_quality to select best face (prioritizes center person + front-facing)
    _, best_face_data = face_detector.score_face_quality(best_frame, best_faces)
    
    if best_face_data is None:
        # Fallback: use largest face
        best_face_data = max(best_faces, key=lambda f: f[2] * f[3] if len(f) >= 4 else 0)
        logger.debug("Using largest face as fallback (quality evaluation failed)")
    
    # Extract bbox from face data
    if len(best_face_data) >= 4:
        x, y, width, height = best_face_data[:4]
    else:
        x, y, width, height = best_face_data
    
    # Add padding (20% on each side)
    padding_x = int(width * 0.2)
    padding_y = int(height * 0.2)
    
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(w, x + width + padding_x)
    y2 = min(h, y + height + padding_y)
    
    # Extract face region
    face_region = best_frame[y1:y2, x1:x2]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), face_region)
    
    if success:
        logger.debug(f"Extracted best face frame: {output_path}")
    else:
        logger.error(f"Failed to save face frame: {output_path}")
    
    return success


def extract_reference_images(
    qualified_clips: List[Tuple[Path, Path]],
    output_folder: Path
) -> List[Tuple[Path, Path, Path, Path]]:
    """
    Extract reference images for all qualified clips.
    
    Args:
        qualified_clips: List of (clip1_path, clip2_path) tuples
        output_folder: Output directory
        
    Returns:
        List of (clip1_path, clip2_path, ref1_path, ref2_path) tuples
        where ref paths are tuples: (first_frame_path, face_frame_path)
    """
    face_detector = FaceDetector()
    results = []
    
    try:
        for clip1_path, clip2_path in qualified_clips:
            logger.info(f"Extracting reference images for: {clip1_path.name}, {clip2_path.name}")
            
            # Extract first frames
            ref1_first = output_folder / f"{clip1_path.stem}_ref_first.jpg"
            ref2_first = output_folder / f"{clip2_path.stem}_ref_first.jpg"
            
            success1 = extract_first_frame(clip1_path, ref1_first)
            success2 = extract_first_frame(clip2_path, ref2_first)
            
            if not success1 or not success2:
                logger.warning(f"Failed to extract first frames for {clip1_path.name}")
                continue
            
            # Extract best face frames
            ref1_face = output_folder / f"{clip1_path.stem}_ref_face.jpg"
            ref2_face = output_folder / f"{clip2_path.stem}_ref_face.jpg"
            
            success1 = extract_best_face_frame(clip1_path, ref1_face, face_detector)
            success2 = extract_best_face_frame(clip2_path, ref2_face, face_detector)
            
            if not success1 or not success2:
                logger.warning(f"Failed to extract face frames for {clip1_path.name}")
                continue
            
            results.append((
                clip1_path,
                clip2_path,
                (ref1_first, ref1_face),
                (ref2_first, ref2_face)
            ))
    
    finally:
        face_detector.cleanup()
    
    logger.info(f"Extracted reference images for {len(results)} clip pairs")
    return results
