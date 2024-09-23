import os
import numpy as np
import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
from itertools import groupby

import torch
import cv2
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor
from utils import extract_video_frames

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "../checkpoints/sam2_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"
IMAGE_HEIGHT = 1920

logger = logging.getLogger(__name__)


def configure_logger(verbosity: int) -> None:
    log_level = logging.INFO  # Default to INFO level
    if verbosity == 0:
        log_level = logging.ERROR
    if verbosity == 2:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


# Represents a point with x and y coordinates
@dataclass
class FramePoint:
    x: int
    y: int


# Represents a player's annotation in a frame
@dataclass
class PlayerAnnotation:
    player_name: str
    point: FramePoint


# Represents a frame with the information of the players on it
@dataclass
class FrameAnnotations:
    frame_number: int
    players_annotations: List[PlayerAnnotation] = field(default_factory=list)

    def add_player_annotation(self, player_annotation: PlayerAnnotation):
        self.players_annotations.append(player_annotation)


# Builds a Dict with all of the frames' annotations from the annotations_data JSON object
def build_frames_annotations_data(
    annotations_data: List[Dict],
) -> Dict[int, FrameAnnotations]:
    frames = {}

    # Offset for y axis
    offset = int(round(-IMAGE_HEIGHT * 0.03))

    # Sort by player_name and player_track[0]
    sorted_annotations = sorted(
        annotations_data, key=lambda x: (x["player_name"], x["player_track"][0])
    )

    # Group by player_name and keep the first entry for each player
    result = []
    for player_name, group in groupby(
        sorted_annotations, key=lambda x: x["player_name"]
    ):
        result.append(next(group))

    # Create a PlayerAnnotation for each player
    for annotation in result:
        player_name = annotation["player_name"]
        frame_number, x, y = annotation["player_track"]

        fixed_y = y + offset
        player_info = PlayerAnnotation(player_name, FramePoint(x, fixed_y))

        if frame_number in frames:
            frames[frame_number].add_player_annotation(player_info)
        else:
            frames[frame_number] = FrameAnnotations(frame_number, [player_info])

    return frames


# Assign an unique integer ID for each player present across all frame annotations
def codify_player_names(
    frame_annotations: Dict[int, FrameAnnotations]
) -> Dict[str, int]:
    player_name_to_id = {}
    current_id = 0

    # Iterate through each frame annotation
    for frame_annotation in frame_annotations.values():
        # Iterate through each player annotation in the current frame
        for player_annotation in frame_annotation.players_annotations:
            player_name = player_annotation.player_name

            # If the player name is not already in the dictionary, assign a new ID
            if player_name not in player_name_to_id:
                player_name_to_id[player_name] = current_id
                current_id += 1

    return player_name_to_id


# Enables mixed-precision computing for faster deep learning
def enable_mixed_precision() -> None:
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# Scan all the JPEG frame names in the images_path directory
def get_frames_names(images_path: Path) -> List[str]:
    frame_names = [
        p
        for p in os.listdir(images_path)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return frame_names


# Saves masks and original frames to designated folders for each frame
def save_masks_and_images(
    video_segments,
    images_path: Path,
    frame_names: List[str],
    player_ids: Dict[str, int],
    output_path: Path,
) -> None:
    # Create the root folder
    root_folder = output_path
    root_folder.mkdir(exist_ok=True)

    # Loop through each frame and its corresponding masks
    for frame_idx in tqdm(video_segments.keys(), desc="Saving masks"):
        frame_masks = video_segments[frame_idx]

        # Create the folder structure for each frame
        frame_folder = root_folder / str(frame_idx)
        video_folder = frame_folder / "video"
        players_folder = frame_folder / "players"

        video_folder.mkdir(parents=True, exist_ok=True)
        players_folder.mkdir(parents=True, exist_ok=True)

        # Save the original video frame
        video_frame_path = images_path / frame_names[frame_idx]
        frame_output_path = video_folder / frame_names[frame_idx]

        original_frame = cv2.imread(str(video_frame_path))
        cv2.imwrite(str(frame_output_path), original_frame)

        # Save each player's mask in the 'players' folder
        for player_id, mask in frame_masks.items():
            mask = np.squeeze(mask)
            mask_image = (mask * 255).astype(np.uint8)
            mask_filename = f"player_{player_id}.png"
            mask_output_path = players_folder / mask_filename
            cv2.imwrite(str(mask_output_path), mask_image)


# Initialize the SAM model with mixed precision
def initialize_sam_model(config: str, checkpoint: str, mixed_precision: bool = True):
    if mixed_precision:
        logger.debug("Enabling mixed precision")
        enable_mixed_precision()

    sam2_model = build_sam2_video_predictor(config, checkpoint)
    logger.info("SAM model initialized successfully")

    return sam2_model


# Process and add points for each player in the frame to the SAM model
def process_video_frames(sam2_model, frames_annotations_data, images_path, frame_names):
    logger.info("Initializing inference state")
    inference_state = sam2_model.init_state(
        video_path=str(images_path)
    )  # init_state doesn't support Path objects

    player_ids = codify_player_names(frames_annotations_data)

    # For label, `1` means positive click and `0` means negative click
    label = np.array([1], np.int32)

    logger.info("Adding player points to the model")
    for frame_number, frame_annotation in frames_annotations_data.items():
        for player_annotation in frame_annotation.players_annotations:
            annotation_object_id = player_ids.get(player_annotation.player_name, None)

            if annotation_object_id is None:
                logger.warning(
                    f"ID not found for player_annotation {player_annotation}"
                )
                continue

            # Add a point (x, y)
            point = np.array(
                [[player_annotation.point.x, player_annotation.point.y]],
                dtype=np.float32,
            )

            logger.debug(
                f"Adding point for {player_annotation.player_name} ID {annotation_object_id} at {point} on frame {frame_number}"
            )

            _, out_obj_ids, out_mask_logits = sam2_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_number,
                obj_id=annotation_object_id,
                points=point,
                labels=label,
            )

    return inference_state, player_ids


# Propagate the points added in one frame throughout the rest of the video
def propagate_masks(sam2_model, inference_state):
    video_segments = {}  # video_segments contains the per-frame segmentation results
    logger.info("Propagating masks throughout the video")
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    logger.info("Mask propagation completed")
    return video_segments


def main():
    parser = argparse.ArgumentParser(description="Video masklet tracker")
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "-a",
        "--annotations_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the annotations",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=False,
        default="layers",
        help="Path of the output masks (./layers)",
    )
    parser.add_argument(
        "-vb",
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Set the logging verbosity: 0=ERROR, 1=INFO, 2=DEBUG (1)",
    )
    args = parser.parse_args()

    configure_logger(args.verbosity)

    video_path = Path(args.video_path)
    annotations_path = Path(args.annotations_path)
    output_path = Path(args.output_path)

    # Path validations
    if not video_path.is_file():
        logger.error(f"Error: The video file {video_path} does not exist.")
        exit(1)

    if not annotations_path.is_file():
        logger.error(f"Error: The JSON file {annotations_path} does not exist.")
        exit(1)

    # Read and parse the JSON file
    with open(annotations_path, "r") as file:
        annotations_data = json.load(file)

    # Build frames' annotations
    frames_annotations_data = build_frames_annotations_data(annotations_data)

    # Extract frames from video
    logger.info(f"Extracting frames from video at {video_path}")
    images_path = Path("images")
    frame_count = extract_video_frames(video_path, images_path)
    logger.info(f"Extracted {frame_count} frames from {video_path} to {images_path}")

    # Scan all the JPEG frame names in this directory
    logger.info(f"Scanning images from {images_path}")
    frame_names = get_frames_names(images_path)
    logger.info(f"{len(frame_names)} images scanned")

    # Initialize the SAM 2 model
    logger.info("Initializing SAM 2 Model")
    sam2_model = initialize_sam_model(CONFIG, CHECKPOINT, mixed_precision=True)

    # Process the video frames and propagate the masks
    inference_state, player_ids = process_video_frames(
        sam2_model, frames_annotations_data, images_path, frame_names
    )
    logger.info("Starting masklet propagation throughout the video")
    video_segments = propagate_masks(sam2_model, inference_state)

    # Save the masks and the original frames
    logger.info(f"Saving all generated masks to output path: {output_path}")
    save_masks_and_images(
        video_segments, images_path, frame_names, player_ids, output_path
    )
    logger.info("Process completed successfully")


if __name__ == "__main__":
    main()
