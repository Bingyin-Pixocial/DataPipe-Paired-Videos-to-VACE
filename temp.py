import argparse
import re
import shutil
import subprocess
import glob
import logging
import sys
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

# VLM imports (optional - will handle import errors gracefully)
try:
    import torch
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
    VLM_AVAILABLE = True
    # Check if accelerate is available for device_map
    try:
        import accelerate
        ACCELERATE_AVAILABLE = True
    except ImportError:
        ACCELERATE_AVAILABLE = False
except ImportError:
    VLM_AVAILABLE = False
    ACCELERATE_AVAILABLE = False
    torch = None
    Qwen3VLMoeForConditionalGeneration = None
    AutoProcessor = None

# Qwen-Image-Edit imports (optional - will handle import errors gracefully)
try:
    from PIL import Image
    from diffusers import QwenImageEditPipeline
    IMAGE_EDIT_AVAILABLE = True
except ImportError:
    IMAGE_EDIT_AVAILABLE = False
    QwenImageEditPipeline = None

STATUS = Enum('STATUS', ('PoseExtraction', 'RefImageGeneration', 'PromptConstruction'))
INIT_STATUS = STATUS.PoseExtraction


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file. If None, only logs to console.
    """
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # More verbose in file
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")
    
    return logger


def get_project_root():
    """Get the project root directory (pixpose folder)."""
    current_file = Path(__file__).resolve()
    return current_file.parent


def run_subprocess_with_logging(cmd, cwd=None, logger=None):
    """
    Run subprocess and log output in real-time.
    
    Args:
        cmd: Command to run (list of strings)
        cwd: Working directory
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        subprocess.CompletedProcess or None if error
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info(f"Executing command: {' '.join(cmd)}")
    if cwd:
        logger.debug(f"Working directory: {cwd}")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                # Remove trailing newline and log
                line = line.rstrip()
                logger.info(f"[SUBPROCESS] {line}")
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info("Command completed successfully")
            return subprocess.CompletedProcess(cmd, process.returncode)
        else:
            logger.error(f"Command failed with return code: {process.returncode}")
            return None
            
    except Exception as e:
        logger.error(f"Error running subprocess: {e}", exc_info=True)
        return None


def run_pose_extraction(conda_env='vace', inputs_dir=None, pose_extractor_dir=None, data_dir=None):
    """
    Run the PoseExtraction stage.
    
    Args:
        conda_env: Name of the conda environment to activate (default: 'vace')
        inputs_dir: Directory containing input videos (default: project_root/inputs)
        pose_extractor_dir: Directory containing pose_extractor (default: project_root/pose_extractor)
        data_dir: Directory to save processed data (default: project_root/data)
    """
    logger = logging.getLogger()
    logger.info("="*70)
    logger.info("Starting PoseExtraction Stage")
    logger.info("="*70)
    
    project_root = get_project_root()
    
    # Set default directories
    if inputs_dir is None:
        inputs_dir = project_root / 'inputs'
    else:
        inputs_dir = Path(inputs_dir)
    
    if pose_extractor_dir is None:
        pose_extractor_dir = project_root / 'pose_extractor'
    else:
        pose_extractor_dir = Path(pose_extractor_dir)
    
    if data_dir is None:
        data_dir = project_root / 'data'
    else:
        data_dir = Path(data_dir)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Inputs directory: {inputs_dir}")
    logger.info(f"Pose extractor directory: {pose_extractor_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Conda environment: {conda_env}")
    
    # Validate directories
    if not inputs_dir.exists():
        logger.error(f"Inputs directory does not exist: {inputs_dir}")
        return False
    
    if not pose_extractor_dir.exists():
        logger.error(f"Pose extractor directory does not exist: {pose_extractor_dir}")
        return False
    
    # Create data folder if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created/verified data directory: {data_dir}")
    
    vace_video_dir = data_dir / 'vace_video'
    vace_video_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created/verified vace_video directory: {vace_video_dir}")
    
    # Find all video files in inputs directory
    logger.info("Scanning for video files...")
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
    video_files = []
    for ext in video_extensions:
        found = glob.glob(str(inputs_dir / ext))
        video_files.extend(found)
        if found:
            logger.debug(f"Found {len(found)} file(s) with extension {ext}")
    
    if not video_files:
        logger.warning(f"No video files found in {inputs_dir}")
        logger.warning(f"Searched for extensions: {', '.join(video_extensions)}")
        return False
    
    video_files = sorted(video_files)
    total_videos = len(video_files)
    logger.info(f"Found {total_videos} video file(s) to process")
    
    # Process each video
    successful = 0
    failed = 0
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).name
        logger.info("")
        logger.info("-"*70)
        logger.info(f"Processing video {idx}/{total_videos}: {video_name}")
        logger.info("-"*70)
        
        # Use absolute path for video
        video_abs_path = Path(video_path).resolve()
        logger.debug(f"Video absolute path: {video_abs_path}")
        
        # Validate video file exists
        if not video_abs_path.exists():
            logger.error(f"Video file does not exist: {video_abs_path}")
            failed += 1
            continue
        
        # Get video file size for logging
        try:
            video_size_mb = video_abs_path.stat().st_size / (1024 * 1024)
            logger.info(f"Video size: {video_size_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not get video file size: {e}")
        
        # Construct the command
        cmd = [
            'conda', 'run', '-n', conda_env,
            'python', 'vace/vace_preproccess.py',
            '--task', 'pose_body_face_hand',
            '--video', str(video_abs_path),
            '--pre_save_dir', str(vace_video_dir),
            '--save_fps', '16'
        ]
        
        logger.info(f"Starting pose extraction for: {video_name}")
        start_time = datetime.now()
        
        # Run the command with logging
        result = run_subprocess_with_logging(
            cmd,
            cwd=str(pose_extractor_dir),
            logger=logger
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result is not None:
            logger.info(f"✓ Successfully processed {video_name} (took {duration:.2f} seconds)")
            successful += 1
        else:
            logger.error(f"✗ Failed to process {video_name}")
            failed += 1
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("PoseExtraction Stage Summary")
    logger.info("="*70)
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("="*70)
    
    return failed == 0


def extract_multiple_frames(video_path, num_frames=10, logger=None):
    """
    Extract multiple frames evenly distributed throughout the video.
    
    Args:
        video_path: Path to the input video file
        num_frames: Number of frames to extract (default: 10)
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        list: List of (frame_number, frame_image) tuples, or empty list if failed
    """
    if logger is None:
        logger = logging.getLogger()
    
    frames = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return frames
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            logger.error(f"Video has no frames: {video_path}")
            cap.release()
            return frames
        
        logger.debug(f"Video has {total_frames} frames at {fps:.2f} fps")
        
        # Calculate frame indices to extract (evenly distributed)
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / (num_frames + 1)
            frame_indices = [int(step * (i + 1)) for i in range(num_frames)]
        
        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frames.append((frame_idx, frame))
            else:
                logger.warning(f"Failed to read frame {frame_idx} from video")
        
        cap.release()
        logger.debug(f"Extracted {len(frames)} frames from video")
        
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}", exc_info=True)
    
    return frames


def evaluate_frame_with_vlm(frame_image, model, processor, logger=None):
    """
    Evaluate a frame using VLM to check if person is facing forward with clear limbs.
    
    Args:
        frame_image: OpenCV image (numpy array)
        model: Qwen3-VL model instance
        processor: AutoProcessor instance
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        dict: Contains 'score' (0-100), 'reasoning', and 'is_good' (bool)
    """
    if logger is None:
        logger = logging.getLogger()
    
    if not VLM_AVAILABLE or model is None or processor is None:
        logger.warning("VLM not available, skipping frame evaluation")
        return {'score': 50, 'reasoning': 'VLM not available', 'is_good': True}
    
    try:
        # Convert BGR to RGB for VLM
        frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        
        # Prepare prompt
        prompt = (
            "Analyze this image of a person. Evaluate:\n"
            "1. Is the person facing forward (towards the camera)?\n"
            "2. Are the person's limbs (arms, legs) clearly visible and not occluded?\n"
            "3. Is the person's pose clear and well-defined?\n\n"
            "Respond with a score from 0-100 where:\n"
            "- 80-100: Excellent - person facing forward, all limbs clearly visible\n"
            "- 60-79: Good - person mostly forward, most limbs visible\n"
            "- 40-59: Fair - person partially forward or some limbs occluded\n"
            "- 0-39: Poor - person facing away or limbs heavily occluded\n\n"
            "Format your response as: 'Score: [number] | Reasoning: [brief explanation]'"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_rgb},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        device = next(model.parameters()).device if hasattr(model, 'parameters') else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        if torch.cuda.is_available() and device.type == "cuda":
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Get input_ids for later trimming
        input_ids = inputs.get('input_ids', None) if isinstance(inputs, dict) else inputs
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
        
        # Decode response - extract only newly generated tokens
        if input_ids is not None:
            # Remove input tokens from generated output
            if isinstance(generated_ids, torch.Tensor):
                generated_ids = generated_ids.cpu()
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.cpu()
            
            # Handle batch dimension
            if input_ids.dim() > 1:
                # Batch processing
                generated_ids_trimmed = [
                    out_ids[len(in_ids):].tolist() 
                    for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
            else:
                # Single sequence
                generated_ids_trimmed = [generated_ids[len(input_ids):].tolist()]
        else:
            # Fallback: decode entire generated sequence
            if isinstance(generated_ids, torch.Tensor):
                generated_ids = generated_ids.cpu()
            generated_ids_trimmed = [generated_ids.tolist()]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse score from response
        score = 50  # default
        reasoning = output_text
        
        # Try to extract score
        score_match = re.search(r'Score:\s*(\d+)', output_text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
            score = max(0, min(100, score))  # Clamp to 0-100
        
        # Extract reasoning if available
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n|$)', output_text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        is_good = score >= 60  # Consider score >= 60 as good
        
        return {
            'score': score,
            'reasoning': reasoning,
            'is_good': is_good,
            'full_response': output_text
        }
        
    except Exception as e:
        logger.error(f"Error evaluating frame with VLM: {e}", exc_info=True)
        return {'score': 50, 'reasoning': f'Error: {str(e)}', 'is_good': True}


def select_best_frame(video_path, num_frames=10, vlm_model=None, vlm_processor=None, vlm_model_name=None, logger=None):
    """
    Extract multiple frames from video, evaluate them with VLM, and select the best one.
    
    Args:
        video_path: Path to the input video file
        num_frames: Number of frames to extract and evaluate (default: 10)
        vlm_model: Qwen3-VL model instance (optional, will load if None and VLM_AVAILABLE)
        vlm_processor: AutoProcessor instance (optional)
        vlm_model_name: Name/path of the VLM model to load (default: "Qwen/Qwen3-VL-30B-A3B-Instruct")
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        tuple: (best_frame_image, best_frame_number, best_score, all_scores) or (None, None, None, [])
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Set default model name if not provided
    if vlm_model_name is None:
        vlm_model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    
    # Extract frames
    frames = extract_multiple_frames(video_path, num_frames, logger)
    
    if not frames:
        logger.error("No frames extracted from video")
        return None, None, None, []
    
    logger.info(f"Evaluating {len(frames)} frames with VLM...")
    
    # Load VLM model if not provided
    model = vlm_model
    processor = vlm_processor
    
    if VLM_AVAILABLE and model is None:
        try:
            logger.info(f"Loading VLM model: {vlm_model_name}...")
            # Prepare loading arguments
            load_kwargs = {
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
            }
            
            # Use device_map only if accelerate is available
            if ACCELERATE_AVAILABLE:
                load_kwargs["device_map"] = "auto"
                logger.debug("Using device_map='auto' (accelerate available)")
            else:
                # Load to CPU or CUDA device manually
                if torch.cuda.is_available():
                    load_kwargs["device"] = "cuda"
                    logger.debug("Using device='cuda' (accelerate not available)")
                else:
                    load_kwargs["device"] = "cpu"
                    logger.debug("Using device='cpu' (accelerate not available)")
            
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                vlm_model_name,
                **load_kwargs
            )
            
            # Move model to device if not using device_map
            if not ACCELERATE_AVAILABLE:
                device = load_kwargs.get("device", "cpu")
                model = model.to(device)
            
            processor = AutoProcessor.from_pretrained(vlm_model_name)
            logger.info(f"✓ VLM model loaded successfully: {vlm_model_name}")
        except Exception as e:
            error_msg = str(e)
            if "accelerate" in error_msg.lower():
                logger.error(f"Failed to load VLM model: accelerate package is required")
                logger.info("Installing accelerate package...")
                try:
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate", "-q"])
                    logger.info("✓ accelerate installed, retrying model load...")
                    # Retry with device_map
                    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                        vlm_model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    )
                    processor = AutoProcessor.from_pretrained(vlm_model_name)
                    logger.info(f"✓ VLM model loaded successfully after installing accelerate: {vlm_model_name}")
                except Exception as retry_error:
                    logger.error(f"Failed to install accelerate or load model: {retry_error}")
                    logger.warning("Falling back to selecting first frame")
                    return frames[0][1], frames[0][0], 50, []
            elif "network" in error_msg.lower() or "connection" in error_msg.lower() or "unreachable" in error_msg.lower():
                logger.error(f"Network error loading VLM model: {error_msg}")
                logger.info("Attempting to load from local cache...")
                try:
                    # Try loading from cache without downloading
                    cache_load_kwargs = {
                        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        "local_files_only": True
                    }
                    if not ACCELERATE_AVAILABLE:
                        if torch.cuda.is_available():
                            cache_load_kwargs["device"] = "cuda"
                        else:
                            cache_load_kwargs["device"] = "cpu"
                    
                    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                        vlm_model_name,
                        **cache_load_kwargs
                    )
                    if not ACCELERATE_AVAILABLE:
                        device = cache_load_kwargs.get("device", "cpu")
                        model = model.to(device)
                    processor = AutoProcessor.from_pretrained(
                        vlm_model_name,
                        local_files_only=True
                    )
                    logger.info(f"✓ VLM model loaded from local cache: {vlm_model_name}")
                except Exception as cache_error:
                    logger.error(f"Model not found in cache: {cache_error}")
                    logger.warning("Falling back to selecting first frame")
                    return frames[0][1], frames[0][0], 50, []
            else:
                logger.error(f"Failed to load VLM model: {e}", exc_info=True)
                logger.warning("Falling back to selecting first frame")
                return frames[0][1], frames[0][0], 50, []
    
    # Evaluate each frame
    frame_scores = []
    for frame_num, frame_image in frames:
        logger.debug(f"Evaluating frame {frame_num}...")
        evaluation = evaluate_frame_with_vlm(frame_image, model, processor, logger)
        
        frame_scores.append({
            'frame_num': frame_num,
            'frame': frame_image,
            'score': evaluation['score'],
            'reasoning': evaluation['reasoning'],
            'is_good': evaluation['is_good']
        })
        
        logger.debug(f"Frame {frame_num}: Score={evaluation['score']}, Good={evaluation['is_good']}")
    
    # Select best frame (highest score)
    if not frame_scores:
        return None, None, None, []
    
    best_frame = max(frame_scores, key=lambda x: x['score'])
    
    logger.info(f"Best frame selected: Frame {best_frame['frame_num']} with score {best_frame['score']}")
    logger.debug(f"Reasoning: {best_frame['reasoning']}")
    
    return (
        best_frame['frame'],
        best_frame['frame_num'],
        best_frame['score'],
        frame_scores
    )


def extract_frame_from_video(video_path, frame_number, logger=None):
    """
    Extract a specific frame from a video by frame number.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame number to extract (0-indexed)
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        numpy array or None: Frame image if successful, None otherwise
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return None
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            return frame
        else:
            logger.warning(f"Failed to read frame {frame_number} from video")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting frame {frame_number} from {video_path}: {e}", exc_info=True)
        return None


def evaluate_face_clarity(frame_image, logger=None):
    """
    Evaluate face clarity in a frame using fast OpenCV-based methods.
    This is time-efficient and doesn't require external models.
    
    Args:
        frame_image: Original frame image (numpy array, BGR format)
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        dict: Contains 'score' (0-100), 'reasoning', 'face_detected' (bool), and 'face_region' (bbox or None)
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        # Convert to grayscale for face detection
        if len(frame_image.shape) == 3:
            gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_image
        
        h, w = gray.shape[:2]
        scores = []
        reasoning_parts = []
        face_detected = False
        face_region = None
        
        # Use OpenCV's built-in Haar cascade face detector (fast and no external files needed)
        # Try to load the cascade classifier
        try:
            # Try multiple possible paths for the cascade file
            cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
            ]
            
            face_cascade = None
            for cascade_path in cascade_paths:
                try:
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    if face_cascade.empty():
                        continue
                    break
                except:
                    continue
            
            if face_cascade is None or face_cascade.empty():
                logger.debug("Face cascade not available, using fallback face detection")
                # Fallback: assume face is in upper center region
                face_region = (int(w*0.3), int(h*0.1), int(w*0.4), int(h*0.3))
                face_detected = True
            else:
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0:
                    # Use the largest face (most likely the main subject)
                    face = max(faces, key=lambda x: x[2] * x[3])  # Largest by area
                    face_region = tuple(face)  # (x, y, width, height)
                    face_detected = True
                else:
                    # No face detected
                    face_detected = False
                    return {
                        'score': 0,
                        'reasoning': 'No face detected',
                        'face_detected': False,
                        'face_region': None
                    }
        except Exception as e:
            logger.debug(f"Face detection error: {e}, using fallback")
            # Fallback: assume face is in upper center region
            face_region = (int(w*0.3), int(h*0.1), int(w*0.4), int(h*0.3))
            face_detected = True
        
        if not face_detected or face_region is None:
            return {
                'score': 0,
                'reasoning': 'No face detected',
                'face_detected': False,
                'face_region': None
            }
        
        x, y, fw, fh = face_region
        
        # Extract face region
        face_roi = gray[y:y+fh, x:x+fw]
        
        if face_roi.size == 0:
            return {
                'score': 0,
                'reasoning': 'Invalid face region',
                'face_detected': False,
                'face_region': None
            }
        
        # 1. Face size score (30 points) - larger faces are clearer
        face_area = fw * fh
        image_area = w * h
        face_ratio = face_area / image_area if image_area > 0 else 0
        # Good face size: 5-15% of image area
        if 0.05 <= face_ratio <= 0.15:
            size_score = 1.0
        elif face_ratio < 0.05:
            size_score = face_ratio / 0.05  # Too small
        else:
            size_score = max(0, 1.0 - (face_ratio - 0.15) / 0.15)  # Too large (might be too close)
        scores.append(size_score * 30)
        reasoning_parts.append(f"FaceSize: {face_ratio:.3f}")
        
        # 2. Face position score (20 points) - centered faces are better
        face_center_x = x + fw / 2
        face_center_y = y + fh / 2
        center_dist_x = abs(face_center_x - w/2) / (w/2) if w > 0 else 1.0
        center_dist_y = abs(face_center_y - h/2) / (h/2) if h > 0 else 1.0
        # Prefer faces in upper-middle region (typical for portrait)
        ideal_y_ratio = 0.25  # Upper 25% of image
        y_dist = abs(face_center_y / h - ideal_y_ratio) if h > 0 else 1.0
        position_score = max(0, 1.0 - (center_dist_x * 0.5 + y_dist * 0.5))
        scores.append(position_score * 20)
        reasoning_parts.append(f"FacePosition: {position_score:.3f}")
        
        # 3. Face sharpness/blur detection (30 points) - Laplacian variance
        # Higher variance = sharper image
        laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        # Good sharpness: 100-500 (varies by image, but this is a reasonable range)
        if 100 <= laplacian_var <= 500:
            sharpness_score = 1.0
        elif laplacian_var < 100:
            sharpness_score = laplacian_var / 100  # Too blurry
        else:
            sharpness_score = max(0, 1.0 - (laplacian_var - 500) / 500)  # Might be over-sharpened/noisy
        scores.append(sharpness_score * 30)
        reasoning_parts.append(f"FaceSharpness: {laplacian_var:.1f}")
        
        # 4. Face region contrast (20 points) - good contrast = clearer features
        face_contrast = np.std(face_roi.astype(float))
        # Good contrast: 20-60
        if 20 <= face_contrast <= 60:
            contrast_score = 1.0
        elif face_contrast < 20:
            contrast_score = face_contrast / 20  # Too low - poor visibility
        else:
            contrast_score = max(0, 1.0 - (face_contrast - 60) / 60)  # Too high - overexposed
        scores.append(contrast_score * 20)
        reasoning_parts.append(f"FaceContrast: {face_contrast:.1f}")
        
        # Calculate final score
        total_score = sum(scores) if scores else 0
        total_score = max(0, min(100, int(total_score)))  # Clamp to 0-100
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Face clarity analysis"
        
        return {
            'score': total_score,
            'reasoning': reasoning,
            'face_detected': True,
            'face_region': face_region
        }
        
    except Exception as e:
        logger.error(f"Error evaluating face clarity: {e}", exc_info=True)
        return {
            'score': 0,
            'reasoning': f'Error: {str(e)}',
            'face_detected': False,
            'face_region': None
        }


def compare_pose_frames(pose_frame, reference_pose=None, logger=None):
    """
    Evaluate a pose skeleton frame to assess if person is facing forward with clear limbs.
    Uses dwpose keypoint definitions and multiple metrics.
    
    Based on dwpose/OpenPose format with 18 body keypoints:
    0: Nose, 1: Neck, 2: Right Shoulder, 3: Right Elbow, 4: Right Wrist,
    5: Left Shoulder, 6: Left Elbow, 7: Left Wrist, 8: Right Hip, 9: Right Knee,
    10: Right Ankle, 11: Left Hip, 12: Left Knee, 13: Left Ankle,
    14: Right Eye, 15: Left Eye, 16: Right Ear, 17: Left Ear
    
    Args:
        pose_frame: Pose skeleton frame image (numpy array)
        reference_pose: Reference pose skeleton image (optional, not used in current implementation)
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        dict: Contains 'score' (0-100), 'reasoning', and 'is_good' (bool)
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        # Convert to grayscale for analysis
        if len(pose_frame.shape) == 3:
            pose_gray = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2GRAY)
        else:
            pose_gray = pose_frame
        
        h, w = pose_gray.shape[:2]
        
        # Calculate multiple quality metrics
        scores = []
        reasoning_parts = []
        
        # 1. Symmetry analysis - forward-facing poses are more symmetric (30 points)
        # Based on dwpose keypoint structure: left/right pairs should be symmetric
        # Key symmetric pairs: shoulders (2,5), elbows (3,6), wrists (4,7), hips (8,11), knees (9,12), ankles (10,13)
        # Split image vertically and compare left/right halves
        mid = w // 2
        left_half = pose_gray[:, :mid]
        right_half = cv2.flip(pose_gray[:, mid:], 1)  # Flip right half to align
        # Resize to same size
        min_width = min(left_half.shape[1], right_half.shape[1])
        if min_width > 0:
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            if left_half.shape == right_half.shape and left_half.size > 0:
                symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
                symmetry_score = max(0, 1.0 - symmetry_diff * 2)  # More symmetric = higher score
                scores.append(symmetry_score * 30)
                reasoning_parts.append(f"Symmetry: {symmetry_score:.3f}")
            else:
                scores.append(0)
                reasoning_parts.append("Symmetry: N/A")
        else:
            scores.append(0)
            reasoning_parts.append("Symmetry: N/A")
        
        # 2. Limb visibility - check for clear limb connections (25 points)
        # Based on dwpose limbSeq connections, check if key limbs are visible
        # Important limbs for forward-facing: shoulders to elbows, elbows to wrists, hips to knees, knees to ankles
        edges_pose = cv2.Canny(pose_gray, 50, 150)
        edge_pixels = np.sum(edges_pose > 0)
        edge_density = edge_pixels / (h * w) if (h * w) > 0 else 0
        # Good pose should have moderate edge density (skeleton lines visible)
        # Target: 0.10-0.20 for good skeleton visibility (based on dwpose skeleton rendering)
        if 0.08 <= edge_density <= 0.25:
            edge_score = 1.0
        elif edge_density < 0.08:
            edge_score = edge_density / 0.08  # Too sparse - limbs not clear
        else:
            edge_score = max(0, 1.0 - (edge_density - 0.20) / 0.20)  # Too dense - noisy
        scores.append(edge_score * 25)
        reasoning_parts.append(f"LimbVisibility: {edge_density:.3f}")
        
        # 3. Pose skeleton density - clear pose should have visible skeleton lines (25 points)
        # Based on dwpose skeleton rendering (uses stickwidth=4, colors, circles for keypoints)
        # Count non-zero pixels (skeleton lines) - threshold for skeleton detection
        skeleton_pixels = np.sum(pose_gray > 20)  # Threshold for skeleton lines (dwpose uses colored lines)
        skeleton_density = skeleton_pixels / (h * w) if (h * w) > 0 else 0
        # Good pose should have reasonable skeleton density (0.05-0.15)
        # This corresponds to having visible body, face, and hand keypoints from dwpose
        if 0.05 <= skeleton_density <= 0.15:
            skeleton_score = 1.0
        elif skeleton_density < 0.05:
            skeleton_score = skeleton_density / 0.05  # Too sparse - keypoints missing
        else:
            skeleton_score = max(0, 1.0 - (skeleton_density - 0.15) / 0.15)  # Too dense - noisy
        scores.append(skeleton_score * 25)
        reasoning_parts.append(f"SkeletonDensity: {skeleton_density:.3f}")
        
        # 4. Center focus - forward-facing person should be centered (10 points)
        # Based on dwpose keypoint structure: person should be in center of frame
        # Calculate center of mass of skeleton (all keypoints combined)
        y_coords, x_coords = np.where(pose_gray > 20)
        if len(x_coords) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            # Distance from image center (normalized)
            center_dist_x = abs(center_x - w/2) / (w/2) if w > 0 else 1.0
            center_dist_y = abs(center_y - h/2) / (h/2) if h > 0 else 1.0
            center_score = max(0, 1.0 - (center_dist_x + center_dist_y) / 2)
            scores.append(center_score * 10)
            reasoning_parts.append(f"CenterFocus: {center_score:.3f}")
        else:
            scores.append(0)
            reasoning_parts.append("CenterFocus: N/A")
        
        # 5. Image quality - clear pose should have good contrast (10 points)
        # Based on dwpose rendering quality (uses colored lines with 0.6 opacity)
        # Calculate contrast (standard deviation of pixel values)
        contrast = np.std(pose_gray.astype(float))
        # Good contrast range: 30-80 (dwpose renders with good contrast)
        if 30 <= contrast <= 80:
            contrast_score = 1.0
        elif contrast < 30:
            contrast_score = contrast / 30  # Too low - poor visibility
        else:
            contrast_score = max(0, 1.0 - (contrast - 80) / 80)  # Too high - overexposed
        scores.append(contrast_score * 10)
        reasoning_parts.append(f"Contrast: {contrast:.1f}")
        
        # Calculate final score
        total_score = sum(scores) if scores else 50
        total_score = max(0, min(100, int(total_score)))  # Clamp to 0-100
        
        is_good = total_score >= 60
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Basic pose analysis"
        
        return {
            'score': total_score,
            'reasoning': reasoning,
            'is_good': is_good
        }
        
    except Exception as e:
        logger.error(f"Error comparing pose frames: {e}", exc_info=True)
        return {'score': 50, 'reasoning': f'Error: {str(e)}', 'is_good': True}


def select_best_frame_by_pose(original_video_path, pose_video_path, num_frames=10, logger=None):
    """
    Select best frame by evaluating ALL frames in the video (not just sampled frames).
    Evaluates each frame to find the one with clearest limbs that faces directly to the camera.
    
    Args:
        original_video_path: Path to original video file
        pose_video_path: Path to pose skeleton video file
        num_frames: Ignored (kept for compatibility, but all frames are evaluated)
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        tuple: (best_frame_image, best_frame_number, best_score, all_scores) or (None, None, None, [])
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        # Open both videos
        original_cap = cv2.VideoCapture(str(original_video_path))
        pose_cap = cv2.VideoCapture(str(pose_video_path))
        
        if not original_cap.isOpened():
            logger.error(f"Failed to open original video: {original_video_path}")
            return None, None, None, []
        
        if not pose_cap.isOpened():
            logger.error(f"Failed to open pose video: {pose_video_path}")
            original_cap.release()
            return None, None, None, []
        
        # Get video properties
        original_total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_total_frames = int(pose_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if original_total_frames == 0 or pose_total_frames == 0:
            logger.error(f"One or both videos have no frames (original: {original_total_frames}, pose: {pose_total_frames})")
            original_cap.release()
            pose_cap.release()
            return None, None, None, []
        
        # Use the minimum frame count to ensure we have matching frames
        total_frames = min(original_total_frames, pose_total_frames)
        
        logger.info(f"Evaluating ALL {total_frames} frames in video (clearest limbs, facing forward, clear face)...")
        
        # Track best frame and all scores
        best_frame_info = None
        best_score = -1
        all_scores = []
        
        # Process all frames sequentially (memory efficient)
        for frame_num in range(total_frames):
            # Extract frame from original video
            original_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret_original, original_frame = original_cap.read()
            
            # Extract corresponding frame from pose video
            pose_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret_pose, pose_frame = pose_cap.read()
            
            if not ret_original or original_frame is None:
                logger.debug(f"Failed to read frame {frame_num} from original video")
                continue
            
            if not ret_pose or pose_frame is None:
                logger.debug(f"Failed to read frame {frame_num} from pose video")
                continue
            
            # Evaluate pose frame
            pose_evaluation = compare_pose_frames(pose_frame, reference_pose=None, logger=logger)
            pose_score = pose_evaluation['score']
            
            # Evaluate face clarity (fast OpenCV-based)
            face_evaluation = evaluate_face_clarity(original_frame, logger=logger)
            face_score = face_evaluation['score']
            
            # Combine scores: 70% pose, 30% face (weighted combination)
            # This ensures we prioritize good pose but also consider face clarity
            combined_score = int(pose_score * 0.7 + face_score * 0.3)
            
            # Store score info
            score_info = {
                'frame_num': frame_num,
                'score': combined_score,
                'pose_score': pose_score,
                'face_score': face_score,
                'pose_reasoning': pose_evaluation['reasoning'],
                'face_reasoning': face_evaluation['reasoning'],
                'face_detected': face_evaluation['face_detected'],
                'is_good': combined_score >= 60  # Combined threshold
            }
            all_scores.append(score_info)
            
            # Update best frame if this is better
            if combined_score > best_score:
                best_score = combined_score
                best_frame_info = {
                    'frame_num': frame_num,
                    'original_frame': original_frame.copy(),  # Copy to preserve frame
                    'pose_frame': pose_frame,
                    'score': combined_score,
                    'pose_score': pose_score,
                    'face_score': face_score,
                    'reasoning': f"Pose: {pose_evaluation['reasoning']}; Face: {face_evaluation['reasoning']}",
                    'is_good': combined_score >= 60
                }
            
            # Log progress periodically (every 10% or every 100 frames, whichever is more frequent)
            if (frame_num + 1) % max(1, min(100, total_frames // 10)) == 0 or (frame_num + 1) == total_frames:
                progress_pct = ((frame_num + 1) / total_frames) * 100
                if best_frame_info is not None:
                    logger.info(f"Progress: {frame_num + 1}/{total_frames} frames ({progress_pct:.1f}%) - Current best: Frame {best_frame_info['frame_num']} (combined={best_score}, pose={best_frame_info['pose_score']}, face={best_frame_info['face_score']})")
                else:
                    logger.info(f"Progress: {frame_num + 1}/{total_frames} frames ({progress_pct:.1f}%)")
        
        # Release video captures
        original_cap.release()
        pose_cap.release()
        
        if best_frame_info is None:
            logger.error("No valid frames found to evaluate")
            return None, None, None, []
        
        logger.info(f"✓ Best frame selected: Frame {best_frame_info['frame_num']} with combined score {best_frame_info['score']} (pose: {best_frame_info['pose_score']}, face: {best_frame_info['face_score']})")
        logger.debug(f"Reasoning: {best_frame_info['reasoning']}")
        
        return (
            best_frame_info['original_frame'],
            best_frame_info['frame_num'],
            best_frame_info['score'],
            all_scores
        )
        
    except Exception as e:
        logger.error(f"Error selecting best frame by pose: {e}", exc_info=True)
        return None, None, None, []


def save_reference_image(frame_image, output_path, logger=None):
    """
    Save a frame image as a reference image.
    
    Args:
        frame_image: OpenCV image (numpy array)
        output_path: Path where the image should be saved
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if logger is None:
        logger = logging.getLogger()
    
    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the frame as JPEG
        success = cv2.imwrite(str(output_path), frame_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            # Get image size for logging
            try:
                img_size_kb = output_path.stat().st_size / 1024
                logger.debug(f"Saved reference image: {output_path.name} ({img_size_kb:.2f} KB)")
            except Exception:
                pass
            return True
        else:
            logger.error(f"Failed to save image to: {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error saving reference image: {e}", exc_info=True)
        return False


def edit_image_with_qwen(frame_image, edit_pipeline=None, model_name="Qwen/Qwen-Image-Edit", prompt=None, logger=None):
    """
    Edit an image using Qwen-Image-Edit model to adjust pose.
    
    Args:
        frame_image: OpenCV image (numpy array, BGR format)
        edit_pipeline: QwenImageEditPipeline instance (optional, will load if None)
        model_name: Name/path of the Qwen-Image-Edit model (default: "Qwen/Qwen-Image-Edit")
        prompt: Edit prompt (default: "Make this character stand still and face forward. Show the entire body with clear limbs.")
        logger: Logger instance (if None, uses default logger)
    
    Returns:
        tuple: (edited_frame_image, success) where edited_frame_image is OpenCV image (BGR) or None if failed
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Set default prompt if not provided
    if prompt is None:
        prompt = "Make this character stand still and face forward. Show the entire body with clear limbs."
    
    if not IMAGE_EDIT_AVAILABLE:
        logger.warning("Qwen-Image-Edit not available. Install diffusers and PIL to enable image editing.")
        logger.warning("Returning original image without editing.")
        return frame_image, False
    
    if not torch:
        logger.warning("PyTorch not available. Cannot use Qwen-Image-Edit.")
        logger.warning("Returning original image without editing.")
        return frame_image, False
    
    try:
        # Convert OpenCV BGR image to PIL RGB image
        frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Load pipeline if not provided
        pipeline = edit_pipeline
        if pipeline is None:
            logger.info(f"Loading Qwen-Image-Edit model: {model_name} (this may take a while on first run)...")
            try:
                pipeline = QwenImageEditPipeline.from_pretrained(model_name)
                logger.info(f"✓ Qwen-Image-Edit model loaded successfully: {model_name}")
                
                # Move pipeline to appropriate device
                if torch.cuda.is_available():
                    pipeline = pipeline.to(torch.bfloat16)
                    pipeline = pipeline.to("cuda")
                    logger.debug("Pipeline moved to CUDA with bfloat16")
                else:
                    pipeline = pipeline.to(torch.float32)
                    pipeline = pipeline.to("cpu")
                    logger.debug("Pipeline moved to CPU with float32")
                
                # Disable progress bar if needed
                pipeline.set_progress_bar_config(disable=None)
            except Exception as e:
                error_msg = str(e)
                if "network" in error_msg.lower() or "connection" in error_msg.lower() or "unreachable" in error_msg.lower():
                    logger.error(f"Network error loading Qwen-Image-Edit model: {error_msg}")
                    logger.info("Attempting to load from local cache...")
                    try:
                        pipeline = QwenImageEditPipeline.from_pretrained(
                            model_name,
                            local_files_only=True
                        )
                        if torch.cuda.is_available():
                            pipeline = pipeline.to(torch.bfloat16)
                            pipeline = pipeline.to("cuda")
                        else:
                            pipeline = pipeline.to(torch.float32)
                            pipeline = pipeline.to("cpu")
                        pipeline.set_progress_bar_config(disable=None)
                        logger.info(f"✓ Qwen-Image-Edit model loaded from local cache: {model_name}")
                    except Exception as cache_error:
                        logger.error(f"Model not found in cache: {cache_error}")
                        logger.warning("Returning original image without editing.")
                        return frame_image, False
                else:
                    logger.error(f"Failed to load Qwen-Image-Edit model: {e}", exc_info=True)
                    logger.warning("Returning original image without editing.")
                    return frame_image, False
        
        # Prepare inputs for the pipeline
        logger.debug(f"Editing image with prompt: {prompt}")
        inputs = {
            "image": pil_image,
            "prompt": prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
        }
        
        # Run inference
        with torch.inference_mode():
            output = pipeline(**inputs)
            edited_pil_image = output.images[0]
        
        # Convert PIL RGB image back to OpenCV BGR image
        edited_rgb = np.array(edited_pil_image)
        edited_bgr = cv2.cvtColor(edited_rgb, cv2.COLOR_RGB2BGR)
        
        logger.debug("✓ Image editing completed successfully")
        return edited_bgr, True
        
    except Exception as e:
        logger.error(f"Error editing image with Qwen-Image-Edit: {e}", exc_info=True)
        logger.warning("Returning original image without editing.")
        return frame_image, False


def run_ref_image_generation(inputs_dir=None, data_dir=None, num_frames=10, use_vlm=True, vlm_model_name=None, use_image_edit=True):
    """
    Run the RefImageGeneration stage.
    Extract multiple frames from each video, evaluate with VLM, and save the best frame as reference image.
    
    Args:
        inputs_dir: Directory containing input videos (default: project_root/inputs)
        data_dir: Directory to save processed data (default: project_root/data)
        num_frames: Number of frames to extract and evaluate per video (default: 10)
        use_vlm: Whether to use VLM for frame evaluation (default: True)
        vlm_model_name: Name/path of the VLM model to use (default: "Qwen/Qwen3-VL-30B-A3B-Instruct")
        use_image_edit: Whether to use Qwen-Image-Edit to adjust pose in best frame (default: True)
    
    Returns:
        bool: True if all videos processed successfully, False otherwise
    """
    logger = logging.getLogger()
    logger.info("="*70)
    logger.info("Starting RefImageGeneration Stage")
    logger.info("="*70)
    
    project_root = get_project_root()
    
    # Set default directories
    if inputs_dir is None:
        inputs_dir = project_root / 'inputs'
    else:
        inputs_dir = Path(inputs_dir)
    
    if data_dir is None:
        data_dir = project_root / 'data'
    else:
        data_dir = Path(data_dir)
    
    # Set default model name if not provided
    if vlm_model_name is None:
        vlm_model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Inputs directory: {inputs_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Number of frames to evaluate per video: {num_frames}")
    logger.info(f"VLM evaluation: {'Enabled' if use_vlm else 'Disabled'}")
    if use_vlm:
        logger.info(f"VLM model: {vlm_model_name}")
    logger.info(f"Image editing: {'Enabled' if use_image_edit else 'Disabled'}")
    
    # Check VLM availability
    if use_vlm and not VLM_AVAILABLE:
        logger.warning("VLM requested but not available. Install transformers and torch to enable VLM evaluation.")
        logger.warning("Falling back to selecting first frame without VLM evaluation.")
        use_vlm = False
    
    # Validate directories
    if not inputs_dir.exists():
        logger.error(f"Inputs directory does not exist: {inputs_dir}")
        return False
    
    # Create output directory
    ref_image_dir = data_dir / 'vace_reference_image'
    ref_image_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created/verified reference image directory: {ref_image_dir}")
    
    # Define VACE video directory for pose-based selection
    vace_video_dir = data_dir / 'vace_video'
    logger.debug(f"VACE video directory: {vace_video_dir}")
    
    # Load VLM model once if using VLM
    vlm_model = None
    vlm_processor = None
    if use_vlm and VLM_AVAILABLE:
        try:
            logger.info(f"Loading VLM model: {vlm_model_name} (this may take a while on first run)...")
            
            # Prepare loading arguments
            load_kwargs = {
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
            }
            
            # Use device_map only if accelerate is available
            if ACCELERATE_AVAILABLE:
                load_kwargs["device_map"] = "auto"
                logger.debug("Using device_map='auto' (accelerate available)")
            else:
                # Load to CPU or CUDA device manually
                if torch.cuda.is_available():
                    load_kwargs["device"] = "cuda"
                    logger.debug("Using device='cuda' (accelerate not available)")
                else:
                    load_kwargs["device"] = "cpu"
                    logger.debug("Using device='cpu' (accelerate not available)")
            
            vlm_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                vlm_model_name,
                **load_kwargs
            )
            
            # Move model to device if not using device_map
            if not ACCELERATE_AVAILABLE:
                device = load_kwargs.get("device", "cpu")
                vlm_model = vlm_model.to(device)
            
            vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
            logger.info(f"✓ VLM model loaded successfully: {vlm_model_name}")
        except Exception as e:
            error_msg = str(e)
            if "accelerate" in error_msg.lower():
                logger.error(f"Failed to load VLM model: accelerate package is required")
                logger.info("Installing accelerate package...")
                try:
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate", "-q"])
                    logger.info("✓ accelerate installed, retrying model load...")
                    # Retry with device_map
                    vlm_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                        vlm_model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    )
                    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
                    logger.info(f"✓ VLM model loaded successfully after installing accelerate: {vlm_model_name}")
                except Exception as retry_error:
                    logger.error(f"Failed to install accelerate or load model: {retry_error}")
                    logger.warning("Falling back to selecting first frame without VLM evaluation.")
                    use_vlm = False
                    vlm_model = None
                    vlm_processor = None
            elif "network" in error_msg.lower() or "connection" in error_msg.lower() or "unreachable" in error_msg.lower():
                logger.error(f"Network error loading VLM model: {error_msg}")
                logger.info("Attempting to load from local cache...")
                try:
                    # Try loading from cache without downloading
                    cache_load_kwargs = {
                        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        "local_files_only": True
                    }
                    if not ACCELERATE_AVAILABLE:
                        if torch.cuda.is_available():
                            cache_load_kwargs["device"] = "cuda"
                        else:
                            cache_load_kwargs["device"] = "cpu"
                    
                    vlm_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                        vlm_model_name,
                        **cache_load_kwargs
                    )
                    if not ACCELERATE_AVAILABLE:
                        device = cache_load_kwargs.get("device", "cpu")
                        vlm_model = vlm_model.to(device)
                    vlm_processor = AutoProcessor.from_pretrained(
                        vlm_model_name,
                        local_files_only=True
                    )
                    logger.info(f"✓ VLM model loaded from local cache: {vlm_model_name}")
                except Exception as cache_error:
                    logger.error(f"Model not found in cache: {cache_error}")
                    logger.warning("Falling back to selecting first frame without VLM evaluation.")
                    use_vlm = False
                    vlm_model = None
                    vlm_processor = None
            else:
                logger.error(f"Failed to load VLM model: {e}", exc_info=True)
                logger.warning("Falling back to selecting first frame without VLM evaluation.")
                use_vlm = False
                vlm_model = None
                vlm_processor = None
    
    # Load Qwen-Image-Edit pipeline once for all videos
    edit_pipeline = None
    if use_image_edit and IMAGE_EDIT_AVAILABLE and torch:
        try:
            logger.info("Loading Qwen-Image-Edit model (this may take a while on first run)...")
            edit_pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
            logger.info("✓ Qwen-Image-Edit model loaded successfully")
            
            # Move pipeline to appropriate device
            if torch.cuda.is_available():
                edit_pipeline = edit_pipeline.to(torch.bfloat16)
                edit_pipeline = edit_pipeline.to("cuda")
                logger.debug("Qwen-Image-Edit pipeline moved to CUDA with bfloat16")
            else:
                edit_pipeline = edit_pipeline.to(torch.float32)
                edit_pipeline = edit_pipeline.to("cpu")
                logger.debug("Qwen-Image-Edit pipeline moved to CPU with float32")
            
            # Disable progress bar if needed
            edit_pipeline.set_progress_bar_config(disable=None)
        except Exception as e:
            error_msg = str(e)
            if "network" in error_msg.lower() or "connection" in error_msg.lower() or "unreachable" in error_msg.lower():
                logger.error(f"Network error loading Qwen-Image-Edit model: {error_msg}")
                logger.info("Attempting to load from local cache...")
                try:
                    edit_pipeline = QwenImageEditPipeline.from_pretrained(
                        "Qwen/Qwen-Image-Edit",
                        local_files_only=True
                    )
                    if torch.cuda.is_available():
                        edit_pipeline = edit_pipeline.to(torch.bfloat16)
                        edit_pipeline = edit_pipeline.to("cuda")
                    else:
                        edit_pipeline = edit_pipeline.to(torch.float32)
                        edit_pipeline = edit_pipeline.to("cpu")
                    edit_pipeline.set_progress_bar_config(disable=None)
                    logger.info("✓ Qwen-Image-Edit model loaded from local cache")
                except Exception as cache_error:
                    logger.error(f"Model not found in cache: {cache_error}")
                    logger.warning("Image editing will be skipped if model is not available.")
                    edit_pipeline = None
            else:
                logger.error(f"Failed to load Qwen-Image-Edit model: {e}", exc_info=True)
                logger.warning("Image editing will be skipped if model is not available.")
                edit_pipeline = None
    else:
        if not use_image_edit:
            logger.info("Image editing disabled by user (--no-image-edit flag).")
        elif not IMAGE_EDIT_AVAILABLE:
            logger.info("Qwen-Image-Edit not available (diffusers/PIL not installed). Image editing will be skipped.")
        elif not torch:
            logger.info("PyTorch not available. Image editing will be skipped.")
    
    # Find all video files in inputs directory
    logger.info("Scanning for video files...")
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
    video_files = []
    for ext in video_extensions:
        found = glob.glob(str(inputs_dir / ext))
        video_files.extend(found)
        if found:
            logger.debug(f"Found {len(found)} file(s) with extension {ext}")
    
    if not video_files:
        logger.warning(f"No video files found in {inputs_dir}")
        logger.warning(f"Searched for extensions: {', '.join(video_extensions)}")
        return False
    
    video_files = sorted(video_files)
    total_videos = len(video_files)
    logger.info(f"Found {total_videos} video file(s) to process")
    
    # Process each video
    successful = 0
    failed = 0
    total_scores = []
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).name
        logger.info("")
        logger.info("-"*70)
        logger.info(f"Processing video {idx}/{total_videos}: {video_name}")
        logger.info("-"*70)
        
        # Use absolute path for video
        video_abs_path = Path(video_path).resolve()
        logger.debug(f"Video absolute path: {video_abs_path}")
        
        # Validate video file exists
        if not video_abs_path.exists():
            logger.error(f"Video file does not exist: {video_abs_path}")
            failed += 1
            continue
        
        # Get video file size for logging
        try:
            video_size_mb = video_abs_path.stat().st_size / (1024 * 1024)
            logger.info(f"Video size: {video_size_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not get video file size: {e}")
        
        # Generate output image name
        video_stem = video_abs_path.stem
        output_image_name = f"{video_stem}_ref.jpg"
        output_image_path = ref_image_dir / output_image_name
        
        logger.info(f"Selecting best frame from: {video_name}")
        logger.debug(f"Output image path: {output_image_path}")
        
        start_time = datetime.now()
        
        # Select best frame using VLM or fallback to first frame
        if use_vlm and vlm_model is not None:
            best_frame, best_frame_num, best_score, all_scores = select_best_frame(
                video_abs_path,
                num_frames=num_frames,
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                vlm_model_name=vlm_model_name,
                logger=logger
            )
            
            if best_frame is not None:
                # Log frame scores summary
                if all_scores:
                    scores_list = [s['score'] for s in all_scores]
                    avg_score = sum(scores_list) / len(scores_list)
                    logger.info(f"Frame evaluation scores: min={min(scores_list)}, max={max(scores_list)}, avg={avg_score:.1f}")
                    total_scores.append(best_score)
                
                # Edit the best frame with Qwen-Image-Edit to adjust pose
                if edit_pipeline is not None:
                    logger.info("Editing best frame with Qwen-Image-Edit to adjust pose...")
                    edited_frame, edit_success = edit_image_with_qwen(
                        best_frame,
                        edit_pipeline=edit_pipeline,
                        logger=logger
                    )
                    if edit_success:
                        logger.info("✓ Image editing completed successfully")
                        best_frame = edited_frame  # Use edited frame for saving
                    else:
                        logger.warning("Image editing failed, using original best frame")
                else:
                    logger.debug("Qwen-Image-Edit pipeline not available, skipping image editing")
                
                # Save the best frame (edited if available, otherwise original)
                success = save_reference_image(best_frame, output_image_path, logger=logger)
            else:
                logger.error("Failed to select best frame")
                success = False
        else:
            # Pose-based frame selection (no VLM)
            logger.info("Using pose-based frame selection (VLM evaluation disabled)")
            
            # Find corresponding pose video
            pose_video_path = find_vace_video(video_name, vace_video_dir)
            
            if pose_video_path and pose_video_path.exists():
                logger.info(f"Found pose video: {pose_video_path.name}")
                
                # Select best frame by comparing pose skeletons
                best_frame, best_frame_num, best_score, all_scores = select_best_frame_by_pose(
                    video_abs_path,
                    pose_video_path,
                    num_frames=num_frames,
                    logger=logger
                )
                
                if best_frame is not None:
                    # Log frame scores summary
                    if all_scores:
                        combined_scores = [s['score'] for s in all_scores]
                        pose_scores = [s.get('pose_score', 0) for s in all_scores]
                        face_scores = [s.get('face_score', 0) for s in all_scores]
                        avg_combined = sum(combined_scores) / len(combined_scores) if combined_scores else 0
                        avg_pose = sum(pose_scores) / len(pose_scores) if pose_scores else 0
                        avg_face = sum(face_scores) / len(face_scores) if face_scores else 0
                        logger.info(f"Combined evaluation scores: min={min(combined_scores)}, max={max(combined_scores)}, avg={avg_combined:.1f}")
                        logger.info(f"  - Pose scores: min={min(pose_scores)}, max={max(pose_scores)}, avg={avg_pose:.1f}")
                        logger.info(f"  - Face scores: min={min(face_scores)}, max={max(face_scores)}, avg={avg_face:.1f}")
                        total_scores.append(best_score)
                    
                    # Edit the best frame with Qwen-Image-Edit to adjust pose
                    if edit_pipeline is not None:
                        logger.info("Editing best frame with Qwen-Image-Edit to adjust pose...")
                        edited_frame, edit_success = edit_image_with_qwen(
                            best_frame,
                            edit_pipeline=edit_pipeline,
                            logger=logger
                        )
                        if edit_success:
                            logger.info("✓ Image editing completed successfully")
                            best_frame = edited_frame  # Use edited frame for saving
                        else:
                            logger.warning("Image editing failed, using original best frame")
                    else:
                        logger.debug("Qwen-Image-Edit pipeline not available, skipping image editing")
                    
                    # Save the best frame (edited if available, otherwise original)
                    success = save_reference_image(best_frame, output_image_path, logger=logger)
                else:
                    logger.error("Failed to select best frame using pose comparison")
                    # Fallback to first frame
                    frames = extract_multiple_frames(video_abs_path, num_frames=1, logger=logger)
                    if frames:
                        fallback_frame = frames[0][1]
                        best_frame_num = frames[0][0]
                        best_score = None
                        
                        # Edit the fallback frame with Qwen-Image-Edit to adjust pose
                        if edit_pipeline is not None:
                            logger.info("Editing fallback frame with Qwen-Image-Edit to adjust pose...")
                            edited_frame, edit_success = edit_image_with_qwen(
                                fallback_frame,
                                edit_pipeline=edit_pipeline,
                                logger=logger
                            )
                            if edit_success:
                                logger.info("✓ Image editing completed successfully")
                                fallback_frame = edited_frame  # Use edited frame for saving
                            else:
                                logger.warning("Image editing failed, using original fallback frame")
                        else:
                            logger.debug("Qwen-Image-Edit pipeline not available, skipping image editing")
                        
                        # Save the fallback frame (edited if available, otherwise original)
                        success = save_reference_image(fallback_frame, output_image_path, logger=logger)
                    else:
                        success = False
                        best_frame_num = None
                        best_score = None
            else:
                logger.warning(f"Pose video not found for {video_name}, using first frame")
                frames = extract_multiple_frames(video_abs_path, num_frames=1, logger=logger)
                if frames:
                    fallback_frame = frames[0][1]
                    best_frame_num = frames[0][0]
                    best_score = None
                    
                    # Edit the fallback frame with Qwen-Image-Edit to adjust pose
                    if edit_pipeline is not None:
                        logger.info("Editing fallback frame with Qwen-Image-Edit to adjust pose...")
                        edited_frame, edit_success = edit_image_with_qwen(
                            fallback_frame,
                            edit_pipeline=edit_pipeline,
                            logger=logger
                        )
                        if edit_success:
                            logger.info("✓ Image editing completed successfully")
                            fallback_frame = edited_frame  # Use edited frame for saving
                        else:
                            logger.warning("Image editing failed, using original fallback frame")
                    else:
                        logger.debug("Qwen-Image-Edit pipeline not available, skipping image editing")
                    
                    # Save the fallback frame (edited if available, otherwise original)
                    success = save_reference_image(fallback_frame, output_image_path, logger=logger)
                else:
                    logger.error("Failed to extract first frame")
                    success = False
                    best_frame_num = None
                    best_score = None
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if success:
            score_info = f" (score: {best_score})" if best_score is not None else ""
            frame_info = f" (frame {best_frame_num})" if best_frame_num is not None else ""
            logger.info(f"✓ Successfully saved reference image: {output_image_name}{frame_info}{score_info} (took {duration:.2f} seconds)")
            successful += 1
        else:
            logger.error(f"✗ Failed to extract reference image from: {video_name}")
            failed += 1
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("RefImageGeneration Stage Summary")
    logger.info("="*70)
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    if total_scores:
        avg_best_score = sum(total_scores) / len(total_scores)
        logger.info(f"Average best frame score: {avg_best_score:.1f}/100")
    logger.info(f"Output directory: {ref_image_dir}")
    logger.info(f"VLM evaluation: {'Used' if use_vlm and vlm_model else 'Not used'}")
    logger.info(f"Image editing: {'Used' if use_image_edit and edit_pipeline else 'Not used'}")
    logger.info("="*70)
    
    return failed == 0


def extract_dance_type(video_name):
    """
    Extract dance type from video filename.
    
    Supports various filename patterns:
    - test_Ballet1_720P.mp4 -> "ballet"
    - train_Samba10_720P.mp4 -> "samba"
    - Ballet1_720P.mp4 -> "ballet"
    - samba_dance_001.mp4 -> "samba"
    
    Args:
        video_name: Video filename (e.g., "test_Ballet1_720P.mp4", "train_Samba10_720P.mp4")
    
    Returns:
        str: Dance type in lowercase (e.g., "ballet", "samba")
    """
    # Extended list of common dance types (case-insensitive)
    dance_types = [
        'ballet', 'hiphop', 'hip-hop', 'hip_hop', 'jazz', 'contemporary',
        'tap', 'ballroom', 'latin', 'salsa', 'tango', 'waltz', 'swing',
        'breakdance', 'break-dance', 'break_dance', 'modern', 'folk',
        'flamenco', 'belly', 'bollywood', 'k-pop', 'kpop', 'samba',
        'cha-cha', 'chacha', 'rumba', 'foxtrot', 'quickstep', 'jive',
        'pasodoble', 'paso-doble', 'paso_doble', 'viennese', 'viennese-waltz',
        'bachata', 'merengue', 'reggaeton', 'cumbia', 'bhangra',
        'capoeira', 'polka', 'mambo', 'charleston'
    ]
    
    video_lower = video_name.lower()
    
    # Strategy 1: Try to find known dance type in the filename (most reliable)
    for dance_type in dance_types:
        # Normalize dance type variations (handle hyphens, underscores, spaces)
        dance_variations = [
            dance_type,
            dance_type.replace('-', '_'),
            dance_type.replace('_', '-'),
            dance_type.replace('-', ''),
            dance_type.replace('_', '')
        ]
        for variation in dance_variations:
            if variation in video_lower:
                return dance_type.lower()
    
    # Strategy 2: Extract from common filename patterns
    # Pattern 1: {prefix}_{DanceType}{number}_{suffix}.{ext}
    # Examples: test_Ballet1_720P.mp4, train_Samba10_720P.mp4
    patterns = [
        r'(?:test|train|val|eval)_([A-Za-z]+)\d+',  # test_Ballet1, train_Samba10
        r'^([A-Za-z]+)\d+_',  # Ballet1_720P (dance type at start)
        r'_([A-Za-z]+)\d+_',  # _Ballet1_ (dance type in middle)
        r'^([A-Za-z]+)_',  # Ballet_720P (dance type at start with underscore)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, video_name, re.IGNORECASE)
        if match:
            potential_dance = match.group(1).lower()
            # Filter out common non-dance words
            excluded_words = ['test', 'train', 'val', 'eval', 'video', 'clip', 'dance', 
                             'mp4', 'avi', 'mov', 'mkv', 'flv', 'src', 'pose', 'body',
                             'face', 'hand', 'ref', 'jpg', 'png', 'jpeg']
            
            # Check if it's a valid dance type (not too short, not excluded)
            if len(potential_dance) >= 3 and potential_dance not in excluded_words:
                # Check if it matches any known dance type (fuzzy match)
                for known_dance in dance_types:
                    if potential_dance in known_dance or known_dance in potential_dance:
                        return known_dance.lower()
                # If no exact match but looks valid, return it
                return potential_dance
    
    # Strategy 3: Try to find capitalized word that looks like a dance type
    # Look for capitalized words (potential dance types)
    capitalized_words = re.findall(r'\b([A-Z][a-z]+)\b', video_name)
    for word in capitalized_words:
        word_lower = word.lower()
        # Check against known dance types
        for dance_type in dance_types:
            if word_lower == dance_type or word_lower in dance_type or dance_type in word_lower:
                return dance_type.lower()
        # If word is reasonably long and not excluded, use it
        excluded_words = ['Test', 'Train', 'Val', 'Eval', 'Video', 'Clip', 'Mp4', 'Avi', 'Mov']
        if len(word) >= 4 and word not in excluded_words:
            return word_lower
    
    # Default fallback
    logger = logging.getLogger()
    logger.warning(f"Could not extract dance type from '{video_name}', using 'dance' as default")
    return 'dance'


def find_vace_video(video_name, vace_video_dir):
    """
    Find the corresponding VACE processed video file.
    
    Args:
        video_name: Original video filename (e.g., "test_Ballet1_720P.mp4")
        vace_video_dir: Directory containing VACE processed videos
    
    Returns:
        Path or None: Path to the VACE video file if found, None otherwise
    """
    video_stem = Path(video_name).stem  # e.g., "test_Ballet1_720P"
    
    # VACE output pattern: {video_name}_src_video-pose_body_face_hand.mp4
    expected_pattern = f"{video_stem}_src_video-pose_body_face_hand.mp4"
    expected_path = vace_video_dir / expected_pattern
    
    if expected_path.exists():
        return expected_path
    
    # Try alternative patterns
    patterns = [
        f"{video_stem}_src_video-pose_body_face_hand.mp4",
        f"{video_stem}_src_video-*.mp4",
        f"*{video_stem}*pose*.mp4",
        f"*{video_stem}*.mp4"
    ]
    
    for pattern in patterns:
        matches = list(vace_video_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def run_prompt_construction(inputs_dir=None, data_dir=None):
    """
    Run the PromptConstruction stage.
    Move inputs folder to data/video, create metadata_vace.csv with video paths and prompts.
    
    Args:
        inputs_dir: Directory containing input videos (default: project_root/inputs)
        data_dir: Directory containing processed data (default: project_root/data)
    
    Returns:
        bool: True if CSV created successfully, False otherwise
    """
    logger = logging.getLogger()
    logger.info("="*70)
    logger.info("Starting PromptConstruction Stage")
    logger.info("="*70)
    
    project_root = get_project_root()
    
    # Set default directories
    if inputs_dir is None:
        inputs_dir = project_root / 'inputs'
    else:
        inputs_dir = Path(inputs_dir)
    
    if data_dir is None:
        data_dir = project_root / 'data'
    else:
        data_dir = Path(data_dir)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Inputs directory: {inputs_dir}")
    logger.info(f"Data directory: {data_dir}")
    
    # Validate directories
    if not inputs_dir.exists():
        logger.error(f"Inputs directory does not exist: {inputs_dir}")
        return False
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    video_dir = data_dir / 'video'  # Destination for moved inputs folder
    vace_video_dir = data_dir / 'vace_video'
    ref_image_dir = data_dir / 'vace_reference_image'
    csv_output_path = data_dir / 'metadata_vace.csv'
    
    logger.info(f"Video directory (destination): {video_dir}")
    logger.info(f"VACE video directory: {vace_video_dir}")
    logger.info(f"Reference image directory: {ref_image_dir}")
    logger.info(f"CSV output path: {csv_output_path}")
    
    # Move inputs folder to data/video
    logger.info("")
    logger.info("Moving inputs folder to data/video...")
    
    # Check if inputs folder is already in the correct location
    if inputs_dir.resolve() == video_dir.resolve():
        logger.info("Inputs folder is already in the correct location (data/video)")
    elif video_dir.exists():
        # If data/video already exists, check if it has content
        logger.warning(f"Video directory already exists: {video_dir}")
        existing_videos = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi')) + \
                        list(video_dir.glob('*.mov')) + list(video_dir.glob('*.mkv')) + \
                        list(video_dir.glob('*.flv'))
        
        if existing_videos:
            logger.info(f"Video directory already contains {len(existing_videos)} video(s), using existing directory")
            # Update inputs_dir to point to video_dir for processing
            inputs_dir = video_dir
        else:
            # video_dir exists but is empty, remove it and move inputs
            logger.info("Video directory exists but is empty, replacing with inputs folder...")
            try:
                video_dir.rmdir()  # Remove empty directory
                shutil.move(str(inputs_dir), str(video_dir))
                logger.info(f"✓ Successfully moved {inputs_dir} to {video_dir}")
                inputs_dir = video_dir
            except Exception as e:
                logger.error(f"Failed to move inputs folder: {e}", exc_info=True)
                return False
    else:
        # Move inputs folder to data/video (rename inputs to video)
        try:
            shutil.move(str(inputs_dir), str(video_dir))
            logger.info(f"✓ Successfully moved {inputs_dir} to {video_dir}")
            # Update inputs_dir to point to the new location
            inputs_dir = video_dir
        except Exception as e:
            logger.error(f"Failed to move inputs folder: {e}", exc_info=True)
            return False
    
    # Validate that required directories exist
    if not vace_video_dir.exists():
        logger.warning(f"VACE video directory does not exist: {vace_video_dir}")
        logger.warning("Make sure PoseExtraction stage has been run.")
    
    if not ref_image_dir.exists():
        logger.warning(f"Reference image directory does not exist: {ref_image_dir}")
        logger.warning("Make sure RefImageGeneration stage has been run.")
    
    # Find all video files in video directory (previously inputs)
    logger.info("")
    logger.info("Scanning for video files...")
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
    video_files = []
    for ext in video_extensions:
        found = glob.glob(str(inputs_dir / ext))
        video_files.extend(found)
        if found:
            logger.debug(f"Found {len(found)} file(s) with extension {ext}")
    
    if not video_files:
        logger.error(f"No video files found in {inputs_dir}")
        return False
    
    video_files = sorted(video_files)
    total_videos = len(video_files)
    logger.info(f"Found {total_videos} video file(s) to process")
    
    # Prepare CSV data
    csv_rows = []
    successful = 0
    missing_vace = 0
    missing_ref = 0
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).name
        video_abs_path = Path(video_path).resolve()
        
        logger.debug(f"Processing {idx}/{total_videos}: {video_name}")
        
        # Get absolute path to video (now in data/video)
        video_path_abs = str(video_abs_path)
        
        # Find corresponding VACE video
        vace_video_path = find_vace_video(video_name, vace_video_dir)
        if vace_video_path:
            vace_video_path_abs = str(vace_video_path.resolve())
        else:
            vace_video_path_abs = ""
            missing_vace += 1
            logger.warning(f"VACE video not found for: {video_name}")
        
        # Find corresponding reference image
        video_stem = video_abs_path.stem
        ref_image_name = f"{video_stem}_ref.jpg"
        ref_image_path = ref_image_dir / ref_image_name
        
        if ref_image_path.exists():
            ref_image_path_abs = str(ref_image_path.resolve())
        else:
            ref_image_path_abs = ""
            missing_ref += 1
            logger.warning(f"Reference image not found for: {video_name}")
        
        # Extract dance type and construct prompt
        dance_type = extract_dance_type(video_name)
        prompt = f'The person is dancing {dance_type}.'
        
        # Add row to CSV data
        csv_rows.append({
            'video': video_path_abs,
            'vace_video': vace_video_path_abs,
            'vace_reference_image': ref_image_path_abs,
            'prompt': prompt
        })
        
        logger.debug(f"  Dance type: {dance_type}")
        logger.debug(f"  Prompt: {prompt}")
        
        if vace_video_path and ref_image_path.exists():
            successful += 1
    
    # Write CSV file manually (without csv module to avoid dependency issues)
    logger.info("")
    logger.info("Writing metadata_vace.csv...")
    
    def escape_csv_field(field):
        """Escape CSV field if it contains comma, quote, or newline."""
        if field is None:
            return ''
        field_str = str(field)
        # If field contains comma, quote, or newline, wrap in quotes and escape quotes
        if ',' in field_str or '"' in field_str or '\n' in field_str:
            return '"' + field_str.replace('"', '""') + '"'
        return field_str
    
    try:
        with open(csv_output_path, 'w', encoding='utf-8') as csvfile:
            # Write header
            fieldnames = ['video', 'vace_video', 'vace_reference_image', 'prompt']
            header_line = ','.join(escape_csv_field(f) for f in fieldnames)
            csvfile.write(header_line + '\n')
            
            # Write data rows
            for row in csv_rows:
                row_values = [
                    escape_csv_field(row.get('video', '')),
                    escape_csv_field(row.get('vace_video', '')),
                    escape_csv_field(row.get('vace_reference_image', '')),
                    escape_csv_field(row.get('prompt', ''))
                ]
                csvfile.write(','.join(row_values) + '\n')
        
        logger.info(f"✓ Successfully created metadata_vace.csv")
        logger.info(f"  Total rows: {len(csv_rows)}")
        logger.info(f"  Complete entries: {successful}")
        logger.info(f"  Missing VACE videos: {missing_vace}")
        logger.info(f"  Missing reference images: {missing_ref}")
        
    except Exception as e:
        logger.error(f"Failed to write CSV file: {e}", exc_info=True)
        return False
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("PromptConstruction Stage Summary")
    logger.info("="*70)
    logger.info(f"Total videos processed: {total_videos}")
    logger.info(f"Complete entries: {successful}")
    logger.info(f"Missing VACE videos: {missing_vace}")
    logger.info(f"Missing reference images: {missing_ref}")
    logger.info(f"CSV file: {csv_output_path}")
    logger.info(f"Final data structure:")
    logger.info(f"  {data_dir}/")
    logger.info(f"    metadata_vace.csv")
    logger.info(f"    video/ (moved from inputs/)")
    logger.info(f"    vace_video/")
    logger.info(f"    vace_reference_image/")
    logger.info("="*70)
    
    return True


def main():
    """Main entry point for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='PixPose preprocessing pipeline')
    parser.add_argument(
        '--stage',
        type=str,
        choices=['PoseExtraction', 'RefImageGeneration', 'PromptConstruction', 'all'],
        default='all',
        help='Which stage to run (default: all)'
    )
    parser.add_argument(
        '--conda-env',
        type=str,
        default='vace',
        help='Conda environment name (default: vace)'
    )
    parser.add_argument(
        '--inputs-dir',
        type=str,
        default=None,
        help='Directory containing input videos (default: project_root/inputs)'
    )
    parser.add_argument(
        '--pose-extractor-dir',
        type=str,
        default=None,
        help='Directory containing pose_extractor (default: project_root/pose_extractor)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory to save processed data (default: project_root/data)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Optional path to log file. If not specified, auto-generates timestamped log file in logs/ directory.'
    )
    parser.add_argument(
        '--no-log-file',
        action='store_true',
        help='Disable automatic log file creation (only log to console)'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=10,
        help='Number of frames to extract and evaluate per video in RefImageGeneration stage (default: 10)'
    )
    parser.add_argument(
        '--no-vlm',
        action='store_true',
        help='Disable VLM evaluation in RefImageGeneration stage (use first frame instead)'
    )
    parser.add_argument(
        '--vlm-model',
        type=str,
        default='Qwen/Qwen3-VL-30B-A3B-Instruct',
        help='VLM model name/path to use for frame evaluation (default: Qwen/Qwen3-VL-30B-A3B-Instruct). Examples: Qwen/Qwen3-VL-7B-Instruct, Qwen/Qwen3-VL-30B-A3B-Instruct'
    )
    parser.add_argument(
        '--no-image-edit',
        action='store_true',
        help='Disable image editing with Qwen-Image-Edit in RefImageGeneration stage (save original best frame without editing)'
    )
    
    args = parser.parse_args()
    
    # Auto-generate log file if not specified and not disabled
    log_file_path = args.log_file
    if not args.no_log_file and log_file_path is None:
        project_root = get_project_root()
        logs_dir = project_root / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stage_suffix = args.stage.lower().replace('all', 'full')
        log_file_path = logs_dir / f'preprocess_{stage_suffix}_{timestamp}.log'
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level=log_level, log_file=log_file_path)
    logger = logging.getLogger()
    
    # Log startup information
    logger.info("")
    logger.info("="*70)
    logger.info("PixPose Preprocessing Pipeline")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Stage(s) to run: {args.stage}")
    logger.info(f"Log level: {args.log_level}")
    if log_file_path:
        logger.info(f"Log file: {log_file_path}")
    logger.info("="*70)
    logger.info("")
    
    pipeline_start_time = datetime.now()
    all_success = True
    stage_results = {}
    
    # Run stages
    if args.stage == 'PoseExtraction' or args.stage == 'all':
        stage_start = datetime.now()
        success = run_pose_extraction(
            conda_env=args.conda_env,
            inputs_dir=args.inputs_dir,
            pose_extractor_dir=args.pose_extractor_dir,
            data_dir=args.data_dir
        )
        stage_end = datetime.now()
        stage_duration = (stage_end - stage_start).total_seconds()
        stage_results['PoseExtraction'] = {
            'success': success,
            'duration': stage_duration
        }
        if not success:
            all_success = False
    
    if args.stage == 'RefImageGeneration' or args.stage == 'all':
        stage_start = datetime.now()
        success = run_ref_image_generation(
            inputs_dir=args.inputs_dir,
            data_dir=args.data_dir,
            num_frames=args.num_frames,
            use_vlm=not args.no_vlm,
            vlm_model_name=args.vlm_model,
            use_image_edit=not args.no_image_edit
        )
        stage_end = datetime.now()
        stage_duration = (stage_end - stage_start).total_seconds()
        stage_results['RefImageGeneration'] = {
            'success': success,
            'duration': stage_duration
        }
        if not success:
            all_success = False
    
    if args.stage == 'PromptConstruction' or args.stage == 'all':
        stage_start = datetime.now()
        success = run_prompt_construction(
            inputs_dir=args.inputs_dir,
            data_dir=args.data_dir
        )
        stage_end = datetime.now()
        stage_duration = (stage_end - stage_start).total_seconds()
        stage_results['PromptConstruction'] = {
            'success': success,
            'duration': stage_duration
        }
        if not success:
            all_success = False
    
    # Final summary
    pipeline_end_time = datetime.now()
    total_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
    
    # Write comprehensive summary
    logger.info("")
    logger.info("")
    logger.info("="*70)
    logger.info("="*70)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*70)
    logger.info("="*70)
    logger.info("")
    logger.info("Execution Information:")
    logger.info(f"  Start time: {pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  End time: {pipeline_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logger.info(f"  Stages executed: {args.stage}")
    logger.info("")
    logger.info("Stage Results:")
    for stage_name, result in stage_results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        duration_str = f"{result['duration']:.2f}s ({result['duration']/60:.2f}min)"
        logger.info(f"  {stage_name:25s} | {status:12s} | Duration: {duration_str}")
    logger.info("")
    logger.info("Overall Status:")
    if all_success:
        logger.info("  ✓ All stages completed successfully")
    else:
        logger.warning("  ✗ Some stages encountered errors")
    logger.info("")
    if log_file_path:
        logger.info(f"Full log saved to: {log_file_path}")
        logger.info("")
    logger.info("="*70)
    logger.info("="*70)
    
    # Also write summary to a separate summary file
    if log_file_path:
        summary_file = log_file_path.parent / f"{log_file_path.stem}_summary.txt"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("PIPELINE EXECUTION SUMMARY\n")
                f.write("="*70 + "\n")
                f.write("\n")
                f.write("Execution Information:\n")
                f.write(f"  Start time: {pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  End time: {pipeline_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n")
                f.write(f"  Stages executed: {args.stage}\n")
                f.write("\n")
                f.write("Stage Results:\n")
                for stage_name, result in stage_results.items():
                    status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
                    duration_str = f"{result['duration']:.2f}s ({result['duration']/60:.2f}min)"
                    f.write(f"  {stage_name:25s} | {status:12s} | Duration: {duration_str}\n")
                f.write("\n")
                f.write("Overall Status:\n")
                if all_success:
                    f.write("  ✓ All stages completed successfully\n")
                else:
                    f.write("  ✗ Some stages encountered errors\n")
                f.write("\n")
                f.write(f"Full log file: {log_file_path}\n")
                f.write("="*70 + "\n")
            logger.info(f"Summary saved to: {summary_file}")
        except Exception as e:
            logger.warning(f"Could not write summary file: {e}")


if __name__ == '__main__':
    main()

