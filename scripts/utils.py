import logging
from pathlib import Path

import cv2


def configure_logger(verbosity: int):
    log_level = logging.INFO  # Default to INFO level
    if verbosity == 0:
        log_level = logging.ERROR
    if verbosity == 2:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    return logging.getLogger(__name__)

# Extract all the frames from video at video_path and
# stores them in images_path
def extract_video_frames(video_path: Path, images_path: Path):
    if not video_path.is_file():
        raise FileNotFoundError(f"The video file {video_path} does not exist.")

    images_path.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save each frame as a .jpg file in the images_path directory
        frame_path = images_path / f"{frame_count:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_count += 1

    # Release the video capture object
    cap.release()
    
    return frame_count