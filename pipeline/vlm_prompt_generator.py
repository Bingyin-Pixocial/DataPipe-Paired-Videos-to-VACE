"""VLM-based prompt generator using Qwen3-VL for analyzing images and videos."""

import logging
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger(__name__)


class VLMPromptGenerator:
    """
    Generate prompts using Qwen3-VL to analyze reference images and control videos.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        use_flash_attention: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize VLM model for prompt generation.
        
        Args:
            model_name: HuggingFace model name for Qwen3-VL
            use_flash_attention: Whether to use flash attention (requires flash-attn package)
            device: Device to use (None = auto-detect)
        """
        self.model_name = model_name
        self.use_flash_attention = use_flash_attention
        self.device = device
        self._model = None
        self._processor = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialize VLM model."""
        if self._initialized:
            return
        
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
            
            logger.info(f"Loading VLM model: {self.model_name}")
            
            # Load model with appropriate settings
            if self.use_flash_attention:
                try:
                    self._model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                        self.model_name,
                        dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map="auto"
                    )
                    logger.info("Using flash_attention_2 for better performance")
                except Exception as e:
                    logger.warning(f"Failed to load with flash_attention_2: {e}. Falling back to default.")
                    self._model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                        self.model_name,
                        dtype="auto",
                        device_map="auto"
                    )
            else:
                self._model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    self.model_name,
                    dtype="auto",
                    device_map="auto"
                )
            
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._initialized = True
            logger.info("VLM model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import transformers or Qwen3-VL: {e}")
            logger.error("Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            raise
    
    def analyze_image(
        self,
        image_path: Path,
        prompt: str = "Describe the camera angle, background, and character appearance in detail."
    ) -> str:
        """
        Analyze an image (first frame) to describe camera angle, background, and character appearance.
        
        Args:
            image_path: Path to the image file
            prompt: Prompt for the VLM to analyze the image
            
        Returns:
            Description string from VLM
        """
        self._initialize()
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.debug(f"Analyzing image: {image_path.name}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path.resolve()),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare for inference
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to appropriate device
        # For models with device_map="auto", transformers handles device placement automatically
        # But we ensure inputs are on the correct device for safety
        device = None
        if hasattr(self._model, 'hf_device_map') and self._model.hf_device_map:
            # Multi-device model - use first device
            first_device = list(self._model.hf_device_map.values())[0]
            if isinstance(first_device, (list, tuple)):
                first_device = first_device[0] if first_device else None
            device = first_device if first_device else None
        else:
            # Single device model - get device from first parameter
            try:
                device = next(self._model.parameters()).device
            except StopIteration:
                # Model has no parameters (shouldn't happen)
                device = None
        
        # Move tensor inputs to device if device is available
        if device:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generate description
        try:
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self._processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            description = output_text[0] if output_text else ""
            if not description:
                logger.warning(f"VLM returned empty description for image: {image_path.name}")
                description = "A character in a video frame."
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            raise
        
        logger.debug(f"Image description: {description[:100]}...")
        return description
    
    def analyze_video(
        self,
        video_path: Path,
        prompt: str = "Describe the concrete movements, poses, and motion patterns in this video in detail."
    ) -> str:
        """
        Analyze a video to describe movements, poses, and motion patterns.
        
        Args:
            video_path: Path to the video file
            prompt: Prompt for the VLM to analyze the video
            
        Returns:
            Description string from VLM
        """
        self._initialize()
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.debug(f"Analyzing video: {video_path.name}")
        
        # Qwen3-VL uses video_url format (as shown in reference implementation)
        # For local files, use file:// URL format
        # file:///absolute/path (three slashes) for absolute paths on Linux/Unix
        # file:///C:/path (Windows absolute path)
        video_path_str = str(video_path.resolve())
        # Ensure proper file:// URL format: file:/// for absolute paths
        if video_path_str.startswith('/'):
            # Linux/Unix absolute path: file:///absolute/path
            video_url = f"file://{video_path_str}"
        elif ':' in video_path_str and video_path_str[1] == ':':
            # Windows absolute path: file:///C:/path
            video_url = f"file:///{video_path_str}"
        else:
            # Relative path (shouldn't happen after resolve(), but handle it)
            video_url = f"file:///{video_path_str}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": video_url,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare for inference
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to appropriate device
        # For models with device_map="auto", transformers handles device placement automatically
        # But we ensure inputs are on the correct device for safety
        device = None
        if hasattr(self._model, 'hf_device_map') and self._model.hf_device_map:
            # Multi-device model - use first device
            first_device = list(self._model.hf_device_map.values())[0]
            if isinstance(first_device, (list, tuple)):
                first_device = first_device[0] if first_device else None
            device = first_device if first_device else None
        else:
            # Single device model - get device from first parameter
            try:
                device = next(self._model.parameters()).device
            except StopIteration:
                # Model has no parameters (shouldn't happen)
                device = None
        
        # Move tensor inputs to device if device is available
        if device:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
        
        # Generate description
        try:
            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self._processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            description = output_text[0] if output_text else ""
            if not description:
                logger.warning(f"VLM returned empty description for video: {video_path.name}")
                description = "A character performing movements."
        except Exception as e:
            logger.error(f"Error generating video description: {e}")
            raise
        
        logger.debug(f"Video description: {description[:100]}...")
        return description
    
    def generate_prompt(
        self,
        reference_image_first: Path,
        vace_control_video: Path
    ) -> str:
        """
        Generate a prompt by combining analysis of reference image and control video.
        
        Args:
            reference_image_first: Path to the first frame reference image
            vace_control_video: Path to the VACE control video
            
        Returns:
            Combined prompt string
        """
        logger.info(f"Generating prompt for: {reference_image_first.name} + {vace_control_video.name}")
        
        # Verify files exist
        reference_image_first = Path(reference_image_first).resolve()
        vace_control_video = Path(vace_control_video).resolve()
        
        if not reference_image_first.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_image_first}")
        if not vace_control_video.exists():
            raise FileNotFoundError(f"Control video not found: {vace_control_video}")
        
        # Analyze reference image (first frame)
        image_prompt = (
            "Describe the camera angle (e.g., front view, side view, overhead), "
            "background setting, and character appearance (clothing, pose, position) in detail."
        )
        try:
            image_description = self.analyze_image(reference_image_first, image_prompt)
        except Exception as e:
            logger.error(f"Failed to analyze image {reference_image_first.name}: {e}")
            image_description = "A character in a video frame."
        
        # Analyze control video (movements)
        video_prompt = (
            "Describe the concrete movements, poses, and motion patterns in this video. "
            "Focus on what the character is doing (e.g., dancing, walking, gesturing) and how they move."
        )
        try:
            video_description = self.analyze_video(vace_control_video, video_prompt)
        except Exception as e:
            logger.error(f"Failed to analyze video {vace_control_video.name}: {e}")
            video_description = "A character performing movements."
        
        # Combine descriptions into a concise prompt
        combined_prompt = f"{image_description.strip()} {video_description.strip()}"
        
        # Clean up and make concise
        # Remove redundant phrases and make it more natural
        combined_prompt = combined_prompt.replace("  ", " ").strip()
        
        # Ensure prompt is not empty
        if not combined_prompt:
            logger.warning("Generated prompt is empty, using fallback")
            combined_prompt = "A character performing movements in a video."
        
        logger.debug(f"Generated prompt: {combined_prompt[:150]}...")
        return combined_prompt
    
    def cleanup(self):
        """Release GPU memory and clean up model resources."""
        import gc
        try:
            if self._model is not None:
                del self._model
                self._model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.debug(f"Error during VLM cleanup: {e}")
