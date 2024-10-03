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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    return frames_names, width, height


def batch_indexes(input_list):
    # Make sure the list is sorted
    sorted_list = sorted(input_list)
    result = {}

    for i, key in enumerate(sorted_list):
        if i == 0:
            previous_key = 0
        else:
            previous_key = sorted_list[i - 1]

        if i == len(sorted_list) - 1:
            next_key = max(sorted_list)
        else:
            next_key = sorted_list[i + 1]

        result[key] = (previous_key, next_key - previous_key)

    return result


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


def generate_masked_video(
    images_path: str | Path, masks_path: str | Path, output_path: str | Path
):
    images_path = Path(images_path)
    masks_path = Path(masks_path)
    output_path = Path(output_path)

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all the image filenames (assume they are named as 00000.jpg, 00001.jpg, etc.)
    image_files = sorted(images_path.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_path}.")

    # Open the first frame to get the video dimensions and FPS (assuming all frames have the same size)
    sample_frame = cv2.imread(str(image_files[0]))
    if sample_frame is None:
        raise ValueError("Failed to read the first frame. Check the image files.")

    height, width, _ = sample_frame.shape
    fps = 25  # Assuming 25 FPS, adjust if needed

    # Define the video writer with mp4 codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise IOError(f"Failed to open VideoWriter with path: {output_path}")

    for image_file in image_files:
        # Read the original image (JPG) and corresponding mask (PNG)
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Warning: Could not read image {image_file}")
            continue

        mask_file = masks_path / f"{image_file.stem}.png"  # Corresponding mask is .png
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Warning: Mask file {mask_file} not found.")
            continue

        # Create a reddish overlay where the mask is not black (non-zero)
        reddish_overlay = np.zeros_like(image)
        reddish_overlay[:, :, 2] = 255  # Full red channel

        # Create a mask where the mask image is non-zero (not black)
        non_zero_mask = mask > 0

        # Blend the reddish overlay with the original image only where the mask is not zero
        blended_image = image.copy()
        blended_image[non_zero_mask] = cv2.addWeighted(
            image[non_zero_mask], 0.7, reddish_overlay[non_zero_mask], 0.3, 0
        )

        # Write the blended frame to the video
        video_writer.write(blended_image)

    # Release the video writer object
    video_writer.release()

    print(f"Masked video saved at {output_path}")
