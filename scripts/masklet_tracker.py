import os
import numpy as np
import argparse
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
from itertools import groupby
from PIL import Image, ImageDraw

import torch
import cv2
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor
from utils import extract_video_frames
from utils import set_up_logger
from utils import batch_indexes
from utils import overlap_masks

import psutil


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "../checkpoints/sam2_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"
IMAGE_HEIGHT = 1920


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


class MaskletTracker:
    def __init__(
        self,
        video_path: str | Path,
        annotations_path: str | Path,
        output_path: str | Path | None = None,
        verbose: int = 0,
    ):
        self.logger = set_up_logger(verbose)
        self.video_path = Path(video_path)
        self.annotations_path = Path(annotations_path)
        self.frames_path = Path("tmp")
        self.output_path = Path(output_path) if output_path else None

        # Read and parse the JSON file
        with open(annotations_path, "r") as file:
            annotations_data = json.load(file)

        self.logger.info("Building frame annotations")
        self.frame_annotations = self._get_frame_annotations_yolo(annotations_data)
        self.player_ids = self._codify_player_names(self.frame_annotations)

        self.logger.info(f"Extracting video frames from: {self.video_path}")
        self.frames_names = extract_video_frames(self.video_path, self.frames_path)
        self.logger.info(f"Extraced {len(self.frames_names)} frames")
        self.logger.info(f"Initializing SAM 2 Model")
        self.sam2_model = self._initialize_sam_model(
            CONFIG, CHECKPOINT, mixed_precision=True
        )
        self.video_segments = dict()
        self.logger.info(f"Model initialized")

    # Masklet tracking
    def track(self):
        self.logger.info("Starting masklet tracking")
        
        self.logger.info("Initializing inference state")
        self.inference_state = self.sam2_model.init_state(
            video_path=str(self.frames_path)
        )  # init_state doesn't support Path objects
        
        self.batch_indexes = batch_indexes(len(self.frames_names), 50)
        
        process = psutil.Process()
        
        for idx, (start_number, max_frames_number) in self.batch_indexes.items():        
            frame_annotations = self.frame_annotations[idx]

            self.inference_state = self._add_frame_points(
                self.sam2_model, self.inference_state, frame_annotations, self.player_ids, self.frames_path
            )

            video_segments = self._propagate_masks(
                self.sam2_model, self.inference_state, start_number, max_frames_number
            )
            
            # Iterate through the frames in the current video_segments
            for frame_idx, obj_masks in video_segments.items():
                # Get the list of masks for all objects in the current frame
                current_masks = list(obj_masks.values())

                if frame_idx in self.video_segments:
                    # Combine existing masks with the current ones using overlap_masks
                    self.logger.debug(f"Overlapping existing frame {frame_idx}")
                    combined_mask = overlap_masks([self.video_segments[frame_idx]] + current_masks)
                    self.video_segments[frame_idx] = combined_mask
                else:
                    # If the frame is not yet in self.video_segments, simply store the combined masks
                    self.logger.debug(f"Overlapping new frame {frame_idx}")
                    self.video_segments[frame_idx] = overlap_masks(current_masks)
                    
            self.sam2_model.reset_state(self.inference_state)
            
            virtual_memory_used_bytes = process.memory_info().vms
            virtual_memory_used_gb = virtual_memory_used_bytes / (1024 ** 3)
            self.logger.debug(f"Virtual memory used: {virtual_memory_used_gb} GB")
        
        if self.output_path is not None:
            self.logger.info(f"Saving masks to: {self.output_path}")
            self._save_masks_and_images(
                self.video_segments,
                self.frames_path,
                self.frames_names,
                self.output_path
            )

        return self.video_segments

    # Draws the YOLO annotation points
    def draw_yolo_annotation_points(self):
        point_radius = 5  # Radius for the annotation points
        point_color = (255, 0, 0)  # Red color for points

        # Iterate over each frame present in self.frame_annotations
        for frame_number, frame_annotations in self.frame_annotations.items():
            # Build the path to the original image
            image_path = Path(self.frames_path) / self.frames_names[frame_number]

            # Open the image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # Iterate over all player annotations in this frame and draw points
            for player_annotation in frame_annotations.players_annotations:
                x, y = player_annotation.point.x, player_annotation.point.y
                draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=point_color)

            # Save the image with drawn points
            output_path = f"frame{frame_number}_ann_points.jpg"
            image.save(output_path)

            self.logger.info(f"Saved annotated frame {frame_number} to {output_path}")


    # Builds a Dict with all of the frames' annotations from the raw_annotations JSON object
    def _get_frame_annotations(
        self, raw_annotations: List[Dict]
    ) -> Dict[int, FrameAnnotations]:
        frames = {}

        # Offset for y axis
        offset = int(round(-IMAGE_HEIGHT * 0.03))

        # Iterate over each player's annotations
        for annotation in raw_annotations:
            player_name = annotation["player_name"]
            player_tracks = annotation["player_track"]

            # Sort the player's tracks by the first element (frame number)
            sorted_tracks = sorted(player_tracks, key=lambda track: track[0])

            # The frame where the player was last seen
            last_frame_number = None

            # Iterate over the sorted tracks to check for gaps
            for track in sorted_tracks:
                frame_number, x, y = track
                frame_number -= 1

                # Check if there is a gap in frames
                if last_frame_number is None or frame_number > last_frame_number + 1:
                    # Apply the offset to y
                    fixed_y = y + offset
                    player_info = PlayerAnnotation(player_name, FramePoint(x, fixed_y))

                    # Add the player's annotation to the frame
                    if frame_number in frames:
                        frames[frame_number].add_player_annotation(player_info)
                    else:
                        frames[frame_number] = FrameAnnotations(
                            frame_number, [player_info]
                        )

                    self.logger.debug(f"Added {player_info} for frame {frame_number}")

                last_frame_number = frame_number

        return frames
    
    # Builds a Dict with all of the frames' annotations from the raw_annotations YOLO JSON object
    def _get_frame_annotations_yolo(self, raw_annotations: List[Dict]) -> Dict[int, FrameAnnotations]:
        frames = {}
        image_width = 1920  # Replace with actual image width if different
        image_height = 1080  # Replace with actual image height if different

        player_id = 0

        # Iterate over each frame's annotations
        for annotation in raw_annotations:
            image_path = annotation["image_path"]
            detections = annotation["detections"]

            # Extract frame number from the image path (subtract 1 to start at 0)
            frame_number = int(image_path.split('/')[-1].replace('.jpg', '')) - 1

            # Iterate over the detections
            for detection in detections:
                if detection["class"] != 0:
                    continue

                bbox = detection["bbox"][0]
                bbox_cx, bbox_cy = bbox[0], bbox[1]

                # Denormalize center x and center y, then round and cast to int
                x = int(round(bbox_cx * image_width))
                y = int(round(bbox_cy * image_height))

                # Create the player annotation with an increasing player name
                player_info = PlayerAnnotation(str(player_id), FramePoint(x, y))

                # Add the player's annotation to the frame
                if frame_number in frames:
                    frames[frame_number].add_player_annotation(player_info)
                else:
                    frames[frame_number] = FrameAnnotations(
                        frame_number, [player_info]
                    )

                player_id += 1
                self.logger.debug(f"Added {player_info} for frame {frame_number}")

        return frames

    # Assign an unique integer ID for each player present across all frame annotations
    def _codify_player_names(
        self, frame_annotations: Dict[int, FrameAnnotations]
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
                    self.logger.debug(
                        f"Assigned ID {current_id} to player name {player_name}"
                    )

                    current_id += 1

        return player_name_to_id

    # Enables mixed-precision computing for faster deep learning
    def _enable_mixed_precision(self) -> None:
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Initialize the SAM model with mixed precision
    def _initialize_sam_model(
        self, config: str, checkpoint: str, mixed_precision: bool = True
    ):
        if mixed_precision:
            self.logger.debug("Enabling mixed precision")
            self._enable_mixed_precision()

        self.logger.debug("Model instantiation")
        sam2_model = build_sam2_video_predictor(config, checkpoint)

        return sam2_model

    # Add points for each player in a given frame to the SAM model
    def _add_frame_points(
        self, sam2_model, inference_state, frame_annotation, player_ids, images_path
    ):
        frame_number = frame_annotation.frame_number
        
        # For label, `1` means positive click and `0` means negative click
        label = np.array([1], np.int32)

        self.logger.info("Adding player points to the model")
        for player_annotation in frame_annotation.players_annotations:
            annotation_object_id = player_ids.get(
                player_annotation.player_name, None
            )

            if annotation_object_id is None:
                self.logger.warning(
                    f"ID not found for player_annotation {player_annotation}"
                )
                continue

            # Add a point (x, y)
            point = np.array(
                [[player_annotation.point.x, player_annotation.point.y]],
                dtype=np.float32,
            )

            self.logger.debug(
                f"Adding point for {player_annotation.player_name} ID {annotation_object_id} at {point} on frame {frame_number}"
            )

            _, out_obj_ids, out_mask_logits = sam2_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_number,
                obj_id=annotation_object_id,
                points=point,
                labels=label,
            )

        return inference_state

    # Propagate the points added in one frame throughout the rest of the video
    def _propagate_masks(self, sam2_model, inference_state, start_number, max_frames_number):
        video_segments = (
            {}
        )  # video_segments contains the per-frame segmentation results
        self.logger.info("Propagating masks throughout the video")
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in sam2_model.propagate_in_video(inference_state, start_number, max_frames_number):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        self.logger.info("Mask propagation completed")
        return video_segments

    # Saves masks and original frames to designated folders for each frame
    def _save_masks_and_images(
        self,
        video_segments,
        images_path: Path,
        frame_names: List[str],
        output_path: Path,
    ) -> None:
        # Create the root 'layers' folder
        root_folder = output_path
        root_folder.mkdir(parents=True, exist_ok=True)

        # Create the 'video' and 'players' folders inside 'layers'
        video_folder = root_folder / "video"
        players_folder = root_folder / "players"
        video_folder.mkdir(parents=True, exist_ok=True)
        players_folder.mkdir(parents=True, exist_ok=True)

        # Loop through each frame and its corresponding masks
        for frame_idx, mask in tqdm(video_segments.items(), desc="Saving masks"):

            # Save the original video frame to the 'video' folder
            frame_name = f"{frame_idx + 1:04d}.jpg"  # Frame name with 4-digit padding
            video_frame_path = images_path / frame_names[frame_idx]
            frame_output_path = video_folder / frame_name

            original_frame = cv2.imread(str(video_frame_path))
            cv2.imwrite(str(frame_output_path), original_frame)

            mask = np.squeeze(mask)
            mask_image = (mask * 255).astype(np.uint8)

            # Save the combined mask image to the 'players' folder
            mask_output_name = f"{frame_idx + 1:04d}.png"
            mask_output_path = players_folder / mask_output_name
            cv2.imwrite(str(mask_output_path), mask_image)


if __name__ == "__main__":
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
        choices=[0, 1, 2, 3],
        default=0,
        help="Set the logging verbosity: 0=No, 1=ERROR, 2=INFO, 3=DEBUG (0)",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    annotations_path = Path(args.annotations_path)
    output_path = Path(args.output_path)
    verbose = args.verbosity

    # Path validations
    if not video_path.is_file():
        logger.error(f"Error: The video file {video_path} does not exist.")
        exit(1)

    if not annotations_path.is_file():
        logger.error(f"Error: The JSON file {annotations_path} does not exist.")
        exit(1)

    masklet_tracker = MaskletTracker(video_path, annotations_path, output_path, verbose)
    # masklet_tracker.draw_annotation_points()
    video_segments = masklet_tracker.track()