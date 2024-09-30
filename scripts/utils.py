import cv2
import logging
import numpy as np
from pathlib import Path
from PIL import Image


def set_up_logger(verbosity: int):
    log_level = logging.CRITICAL + 1  # No log
    if verbosity == 1:
        log_level = logging.ERROR
    elif verbosity == 2:
        log_level = logging.INFO
    elif verbosity == 3:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(__name__)

# Extract all the frames from video at video_path and stores them in images_path
def extract_video_frames(video_path: str | Path, images_path: str | Path):
    video_path = Path(video_path)
    images_path = Path(images_path)
    
    if not video_path.is_file():
        raise FileNotFoundError(f"The video file {video_path} does not exist.")

    images_path.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    frames_names = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save each frame as a .jpg file in the images_path directory
        frame_name = f"{frame_count:05d}.jpg"
        frame_path = images_path / frame_name
        cv2.imwrite(str(frame_path), frame)
        frames_names.append(frame_name)
        frame_count += 1

    # Release the video capture object
    cap.release()
    
    return frames_names


def batch_indexes(total_frames: int, annotation_interval: int):
    indexes_dict = {}
    
    # (start_frame, max_number_frames)
    indexes_dict[0] = (0, annotation_interval)
    
    previous_frame = 0
    total = annotation_interval

    while total <= total_frames:
        actual_frame = total - 1
        max_frame_number = actual_frame + annotation_interval - previous_frame
        indexes_dict[actual_frame] = (previous_frame, max_frame_number)
        previous_frame = actual_frame
        total += annotation_interval

    return indexes_dict

# Overlaps a list of np.arrays using logical OR
def overlap_masks(masks: list):
    if not masks:
        raise ValueError("Mask list is empty")
    
    # Initialize combined_mask with the first mask
    combined_mask = masks[0]

    # Ensure all masks have the same shape as the first one
    for mask in masks[1:]:
        if mask.shape != combined_mask.shape:
            raise ValueError("All masks must have the same shape to overlap")

        # Apply logical OR entry by entry
        combined_mask = np.logical_or(combined_mask, mask)
    
    return combined_mask
