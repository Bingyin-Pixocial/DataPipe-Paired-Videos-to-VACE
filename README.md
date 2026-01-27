# Paired Video to VACE Data Processing Pipeline

This pipeline processes paired videos through 6 sequential stages to produce VACE training data.

## Overview

The pipeline takes paired videos (2 videos per folder) and processes them through:
1. **Mirrored video correction** (Stage 1): Flips horizontally any videos with "mirrored" in filename
2. **FPS normalization** (Stage 2): Converts all videos to target FPS (default: 16 fps) for consistency
3. **Clip extraction** (Stage 3): Cuts videos into clips using sliding window (num_frames, num_stride)
4. **Quality check and screening** (Stage 4): Uses motion similarity and DTW-based similarity scores (from motion similarity notebook), moves unqualified clips to `unqualified/`
5. **Reference image extraction** (Stage 5): Extracts first frame and best face frame for each clip
6. **Metadata construction** (Stage 6): Creates `metadata_vace.csv` with training samples

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py \
    --root_folder /path/to/paired/videos \
    --output_folder /path/to/output \
    --num_frames 81 \
    --num_stride 30 \
    --motion_similarity_threshold 0.8 \
    --dtw_similarity_threshold 0.9 \
    --max_video_length 15.0
```

### Arguments

- `--root_folder`: Root folder containing paired videos in subfolders (required)
- `--output_folder`: Output folder for processed clips and metadata (required)
- `--num_frames`: Number of frames per clip (default: 81)
- `--num_stride`: Stride between clip starting frames (default: 30)
- `--motion_similarity_threshold`: Minimum motion similarity score threshold (default: 0.8)
- `--dtw_similarity_threshold`: Minimum DTW-based similarity score threshold (default: 0.9)
- `--fps`: Target FPS for processing (default: 16.0)
- `--max_video_length`: Maximum video length in seconds. Videos longer than this will be trimmed from the end (default: 15.0)
- `--log_level`: Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
- `--skip_stages`: Skip specific stages (0, 1, 2, 3-6), e.g., `--skip_stages 0 2 3` (use 0 to skip video trimming, 2 to skip FPS normalization)

## Folder Structure

The pipeline expects paired videos in the following structure:
```
root_folder/
  ├── subfolder1/
  │   ├── video1.mp4
  │   └── video2.mp4
  ├── subfolder2/
  │   ├── video3.mp4
  │   └── video4.mp4
  └── ...
```

The pipeline will recursively find lowest-level directories containing exactly 2 video files.

## Output Structure

```
output_folder/
  ├── pipeline.log      # Complete pipeline execution log
  ├── metadata_vace.csv  # Global metadata file (all paths relative to this folder)
  ├── quality_scores.csv # Quality check scores for all clips (Stage 4)
  ├── trimmed/          # Trimmed videos (Stage 0)
  ├── mirror_corrected/  # Mirrored videos corrected (Stage 1)
  ├── fps_normalized/   # FPS-normalized videos (Stage 2)
  └── clips/            # Extracted video clips (Stage 3)
      ├── clip1_a.mp4
      ├── clip1_b.mp4
      ├── clip2_a.mp4
      ├── clip2_b.mp4
      ├── clip1_a_ref_first.jpg
      ├── clip1_a_ref_face.jpg
      ├── clip1_b_ref_first.jpg
      ├── clip1_b_ref_face.jpg
      └── unqualified/   # Unqualified clips moved here
```

**Note**: The original dataset remains completely untouched. All processing happens in the output folder. All logs are saved to `pipeline.log` in the output folder.

## Metadata Format

The `metadata_vace.csv` file contains training samples with columns:
- `video`: Path to source video clip
- `vace_video`: Path to target VACE video clip
- `vace_reference_image`: JSON array with reference image paths (first frame + face frame)
- `prompt`: Training prompt text

For each clip pair (clip1_a, clip1_b), two training samples are created:
1. `video: clip1_a.mp4, vace_video: clip1_b.mp4, vace_reference_image: [clip1_a_ref_first.jpg, clip1_a_ref_face.jpg]`
2. `video: clip1_b.mp4, vace_video: clip1_a.mp4, vace_reference_image: [clip1_b_ref_first.jpg, clip1_b_ref_face.jpg]`

## Dependencies

- OpenCV (cv2) for video processing and optical flow
- NumPy for array operations
- InsightFace for face detection (same as WAN-Animate) - preferred method
- MediaPipe or OpenCV DNN for face detection - fallback methods
- ffmpeg for video manipulation

## Pipeline Stages

### Stage 0: Video Trimming
- Trims videos longer than `max_video_length` (default: 15 seconds) by removing the end portion
- Videos shorter than `max_video_length` are kept as-is (no trimming)
- Example: 16s video → trimmed to 15s (removes last 1s), 25s video → trimmed to 15s (removes last 10s)
- Uses FFmpeg `-t` flag to set duration from start (effectively trims from end)
- Uses stream copy (`-c copy`) for fast processing without re-encoding
- Saves trimmed videos to `output_folder/trimmed/` (preserves original dataset)
- Ensures all videos are ≤15s before further processing

### Stage 1: Mirrored Video Correction
- Detects videos with "mirrored" in filename (case-insensitive)
- Flips videos horizontally using `ffmpeg -vf hflip`
- Saves corrected videos to `output_folder/mirror_corrected/` (preserves original dataset)
- Copies non-mirrored videos to output folder as well

### Stage 2: FPS Normalization
- Converts all videos to target FPS (default: 16 fps) while preserving all frames
- Keeps all original frames and extends duration accordingly
- Example: 30 fps video with 150 frames (5s) → 16 fps video with 150 frames (9.375s)
- Saves normalized videos to `output_folder/fps_normalized/` (preserves original dataset)
- Skips videos already at target FPS (within 0.01 tolerance)
- Ensures consistent frame rates for accurate frame calculations

### Stage 3: Clip Extraction
- Cuts videos into clips using sliding window
- Handles corner cases (stride > length, short videos, etc.)
- Names clips uniquely: `clip1_a`, `clip1_b`, `clip2_a`, `clip2_b`, etc.
- Uses re-encoding (not `-c copy`) for frame-accurate extraction

### Stage 4: Quality Check and Screening
- Uses **both motion similarity and DTW-based similarity** for quality assessment (from motion similarity notebook)
  - **Motion similarity score**: Pattern alignment of motion magnitudes using optical flow (default threshold: 0.8)
    - Extracts motion features using optical flow (cv2.calcOpticalFlowFarneback)
    - Uses path straightness (40%), pattern correlation (30%), distance similarity (20%), angle similarity (10%)
    - Measures how well motion patterns align visually
  - **DTW-based score**: Overall movement similarity using Dynamic Time Warping with path straightness (default threshold: 0.9)
    - Uses path straightness as primary metric (95% weight)
    - Measures how well motion sequences align temporally
    - Handles temporal misalignment and different movement speeds
  - **Both scores must pass** their thresholds for a clip pair to be qualified
  - **Multi-person detection**: Uses YOLO + ByteTrack to detect multiple people in clips
    - For multi-person clips, segments the center person using SAM (Segment Anything Model)
    - Tracks the center person across frames using IoU-based matching
    - Computes motion similarity and DTW scores for the segmented person only
    - Original clips are preserved (segmented videos are temporary and deleted after quality check)
  - Logs detailed statistics: motion similarity score and DTW similarity score
  - Saves all scores (qualified and unqualified) to `quality_scores.csv` for data analysis
    - CSV columns: clip_a_name, clip_b_name, motion_similarity_score, dtw_score, average_score, sum_score
  - Moves unqualified clips to `unqualified/` subdirectory (not deleted)

### Stage 5: Reference Image Extraction
- Extracts first frame from each qualified clip
- Finds best face frame using InsightFace (same as WAN-Animate) with clarity evaluation
  - Uses InsightFace with multiple resolution detectors (640x640, 320x320, 160x160)
  - Evaluates face clarity based on size, position, sharpness (Laplacian variance), and contrast
  - Selects frame with highest clarity score
  - Falls back to MediaPipe or OpenCV DNN if InsightFace unavailable
- Handles multi-person videos by selecting largest/clearest face
- Saves as `{clip_name}_ref_first.jpg` and `{clip_name}_ref_face.jpg`

### Stage 6: Metadata Construction
- Creates `metadata_vace.csv` in root output folder
- Generates two training samples per clip pair (bidirectional)
- Formats reference images as JSON array string
- All paths are relative to output folder root

## Notes

- **Original dataset is never modified** - all processing happens in the output folder
- **Video trimming** (Stage 0) ensures all videos are ≤15s before further processing
  - Videos longer than max_video_length are trimmed from the end
  - Videos shorter than max_video_length are kept unchanged
- Mirrored videos are detected by filename containing "mirrored" (case-insensitive)
- FPS normalization preserves all original frames and extends duration accordingly
- Quality check uses both motion similarity and DTW-based similarity scores (from motion similarity notebook)
  - Both scores must pass their thresholds (default: motion >= 0.8 AND dtw >= 0.9)
  - Motion similarity: uses optical flow to extract motion features, then DTW path analysis
  - DTW-based score: uses path straightness method (95% weight) + distance similarity (5% weight)
  - Measures temporal alignment of motion sequences using Dynamic Time Warping
  - **Multi-person handling**: Detects multiple people using YOLO, segments center person using SAM
    - Motion features are extracted from segmented videos (for accurate comparison)
    - Original clips are preserved (segmented videos are temporary)
- Quality check logs detailed statistics for each clip pair
- Quality scores are saved to `quality_scores.csv` for all clips (qualified and unqualified)
- Unqualified clips are moved to `unqualified/` subdirectory (not deleted)
- Face detection uses InsightFace (same as WAN-Animate) with clarity evaluation for best quality
- Face detection handles multi-person videos by selecting largest/clearest face
- All video paths are resolved to absolute paths to avoid path issues
