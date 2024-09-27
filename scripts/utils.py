import cv2
import logging
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

def batch_indexes(total_indices):
    indexes_list = []
    list_size = 100
    indexes_list.append(list(range(0, 50)))
    start = 50

    while start <= total_indices:
        end = start + list_size
        sublist = list(range(start, min(end, total_indices + 1)))
        indexes_list.append(sublist)
        start = end - 50

    return indexes_list

def create_overlap_image(image1_path, image2_path, output_path):
    mask1 = Image.open(image1_path).convert('L')
    mask2 = Image.open(image2_path).convert('L')

    overlapped = Image.new('L', mask1.size)
    for x in range(mask1.width):
        for y in range(mask1.height):
            pixel1 = mask1.getpixel((x, y))
            pixel2 = mask2.getpixel((x, y))
            overlapped.putpixel((x, y), pixel1 | pixel2)
    overlapped.save(output_path)
