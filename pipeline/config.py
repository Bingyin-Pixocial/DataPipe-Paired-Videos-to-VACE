"""Configuration for paired video to VACE pipeline."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for the paired video to VACE pipeline."""
    
    root_folder: str
    output_folder: str
    num_frames: int = 81
    num_stride: int = 30
    motion_similarity_threshold: float = 0.8  # Minimum motion similarity score (default: 0.8)
    dtw_similarity_threshold: float = 0.9  # Minimum DTW-based similarity score (default: 0.9)
    fps: float = 16.0
    max_video_length: float = 15.0  # Maximum video length in seconds (default: 15.0)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.motion_similarity_threshold < 0 or self.motion_similarity_threshold > 1:
            raise ValueError("motion_similarity_threshold must be between 0 and 1")
        if self.dtw_similarity_threshold < 0 or self.dtw_similarity_threshold > 1:
            raise ValueError("dtw_similarity_threshold must be between 0 and 1")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.num_stride <= 0:
            raise ValueError("num_stride must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.max_video_length <= 0:
            raise ValueError("max_video_length must be positive")
