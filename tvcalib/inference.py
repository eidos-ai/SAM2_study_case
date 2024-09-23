import cv2
import json
import kornia
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from tqdm import tqdm
from matplotlib.patches import Polygon
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from dataclasses import dataclass
from pathlib import Path
from SoccerNet.Evaluation.utils_calibration import SoccerPitch
from tvcalib.cam_modules import SNProjectiveCamera
from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import generate_class_synthesis, get_line_extremities
from tvcalib.sncalib_dataset import custom_list_collate
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from tvcalib.inference import InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel
from tvcalib.inference import get_camera_from_per_sample_output
from tvcalib.utils import visualization_mpl_min as viz


@dataclass
class WorldCoords:
    x: float
    y: float


class VideoCalibration:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.images_path = Path("tmp/raw_frames")
        self.json_tags_path = Path("data/soccernet/players.json")
        self.output_dir = Path("tmp")
        self.checkpoint = "data/segment_localization/train_59.pt"
        self.gpu = True
        self.nworkers = 8
        self.batch_size_seg = 8
        self.batch_size_calib = 256
        self.image_width = 1920
        self.image_height = 1080
        self.optim_steps = 2000
        self.lens_dist = False
        self.write_masks = False
        self.field_width = 68
        self.field_height = 105
        self.radius = 1
        self.df = None
        self.object3dcpu = None
        self.target_scale = 5
        self.players_tags = {}
        with open(self.json_tags_path, 'r') as file:
            tags = json.load(file)
        for player in tags:
            for track in player["player_track"]:
                frame_name = f"{int(track[0]):04d}.jpg"
                if frame_name not in self.players_tags:
                    self.players_tags[frame_name] = []
                self.players_tags[frame_name].append({player["player_name"]: [track[1], track[2]]})
        self.players_tags = dict(sorted(self.players_tags.items()))

    def pixel_to_world_coords(self, pixel_coords, img_width, img_height):
        x_scale = self.field_height / img_width
        y_scale = self.field_width / img_height
        x_meters = pixel_coords[0] * x_scale
        y_meters = pixel_coords[1] * y_scale
        x_field = (x_meters - (self.field_height / 2)).numpy()
        y_field = (y_meters - (self.field_width / 2)).numpy()
        return WorldCoords(x=round(x_field[0], 2), y=round(y_field[0], 2))

    def extract_frames(self):
        self.images_path.mkdir(parents=True, exist_ok=True)
        output_frames_path = self.output_dir / "frames"
        output_frames_path.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(self.video_path))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = self.images_path / f"{frame_count + 1:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1
        cap.release()

    def segment_frames(self):
        device = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"

        object3d = SoccerPitchLineCircleSegments(
            device=device, base_field=SoccerPitchSNCircleCentralSplit()
        )
        object3dcpu = SoccerPitchLineCircleSegments(
            device="cpu", base_field=SoccerPitchSNCircleCentralSplit()
        )

        lines_palette = [0, 0, 0]
        for line_class in SoccerPitch.lines_classes:
            lines_palette.extend(SoccerPitch.palette[line_class])

        fn_generate_class_synthesis = partial(generate_class_synthesis, radius=4)
        fn_get_line_extremities = partial(get_line_extremities, maxdist=30, width=455, height=256, num_points_lines=4, num_points_circles=8)

        dataset_seg = InferenceDatasetSegmentation(self.images_path, self.image_width, self.image_height)
        dataloader_seg = torch.utils.data.DataLoader(
            dataset_seg,
            batch_size=self.batch_size_seg,
            num_workers=self.nworkers,
            shuffle=False,
            collate_fn=custom_list_collate,
        )
        model_seg = InferenceSegmentationModel(self.checkpoint, device)

        image_ids = []
        keypoints_raw = []
        (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)
        for batch_dict in tqdm(dataloader_seg):
            with torch.no_grad():
                sem_lines = model_seg.inference(batch_dict["image"].to(device))
            sem_lines = sem_lines.cpu().numpy().astype(np.uint8)

            with Pool(self.nworkers) as p:
                skeletons_batch = p.map(fn_generate_class_synthesis, sem_lines)
                keypoints_raw_batch = p.map(fn_get_line_extremities, skeletons_batch)

            if self.write_masks:
                for image_id, mask in zip(batch_dict["image_id"], sem_lines):
                    mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
                    mask.putpalette(lines_palette)
                    mask.convert("RGB").save(self.output_dir / "masks" / image_id)

            image_ids.extend(batch_dict["image_id"])
            keypoints_raw.extend(keypoints_raw_batch)

        self.image_ids = image_ids
        self.keypoints_raw = keypoints_raw
        self.object3d = object3d
        self.object3dcpu = object3dcpu

    def calibrate_camera(self):
        device = "cuda" if self.gpu and torch.cuda.is_available() else "cpu"
        model_calib = TVCalibModule(
            self.object3d,
            get_cam_distr(1.96, self.batch_size_calib, 1),
            get_dist_distr(self.batch_size_calib, 1) if self.lens_dist else None,
            (self.image_height, self.image_width),
            self.optim_steps,
            device,
            log_per_step=False,
            tqdm_kwqargs=None,
        )

        dataset_calib = InferenceDatasetCalibration(self.keypoints_raw, self.image_width, self.image_height, self.object3d)
        dataloader_calib = torch.utils.data.DataLoader(dataset_calib, self.batch_size_calib, collate_fn=custom_list_collate)

        per_sample_output = defaultdict(list)
        per_sample_output["image_id"] = [[x] for x in self.image_ids]
        for x_dict in dataloader_calib:
            _batch_size = x_dict["lines__ndc_projected_selection_shuffled"].shape[0]

            points_line = x_dict["lines__px_projected_selection_shuffled"]
            points_circle = x_dict["circles__px_projected_selection_shuffled"]

            per_sample_loss, cam, _ = model_calib.self_optim_batch(x_dict)
            output_dict = tensor2list(detach_dict({**cam.get_parameters(_batch_size), **per_sample_loss}))

            output_dict["points_line"] = points_line
            output_dict["points_circle"] =  points_circle
            for k in output_dict.keys():
                per_sample_output[k].extend(output_dict[k])

        df = pd.DataFrame.from_dict(per_sample_output)
        df = df.explode(column=[k for k, v in per_sample_output.items() if isinstance(v, list)])
        df.set_index("image_id", inplace=True, drop=False)
        self.df = df

        dict_list = []
        for _, row in self.df.iterrows():
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if isinstance(value, torch.Tensor):
                    row_dict[key] = value.numpy().tolist()
                elif isinstance(value, np.ndarray):
                    row_dict[key] = value.tolist()
            image_id = row_dict.pop("image_id")
            if "checkpoint" in image_id:
                continue
            dict_list.append({image_id: row_dict})

        with open(f"{self.output_dir}/metadata.json", 'w') as json_file:
            json.dump(dict_list, json_file, indent=4)

    def get_world_coordinates(self, image_id: str, players_tags): # players_tags: [..., {"player1": [x,y]}, ...]
        image = Image.open(self.images_path / image_id).convert("RGB")
        image = T.functional.to_tensor(image)
        h = torch.tensor(self.df[self.df["image_id"] == image_id]["homography"].values[0])

        S = torch.eye(3).unsqueeze(0)
        S[:, 0, 0] = S[:, 1, 1] = self.target_scale
        T_adjusted = torch.eye(3).unsqueeze(0)
        T_adjusted[:, 0, -1] = self.field_height / 2
        T_adjusted[:, 1, -1] = self.field_width / 2

        def _warp_img(image, H):
            warped_top = kornia.geometry.transform.homography_warp(
                image.unsqueeze(0),
                S @ T_adjusted @ H,
                dsize=(int(self.field_width * self.target_scale), int(self.field_height * self.target_scale)),
                normalized_homography=False,
                normalized_coordinates=False,
                mode="nearest",
            ).squeeze()
            return warped_top

        mask = _warp_img(image, h)
        mask = torch.permute(mask, (1, 2, 0))
        mask = (mask * 255).to(torch.uint8)
        mask = 255 - mask

        ftemplate = Path("template_pitch_t.png") 
        template = T.functional.to_tensor(Image.open(ftemplate).convert("RGB"))
        template = T.functional.resize(template, (int(self.target_scale*self.field_width), int(self.target_scale*self.field_height)))
        template = torch.permute(template, (1, 2, 0))
        template = (template * 255).to(torch.uint8)
        background = Image.fromarray(template.numpy())
        foreground = Image.fromarray(mask.numpy())
        blended = Image.blend(background, foreground, alpha=0.5)

        
        world_coords_list = []
        for player_tag in players_tags:
            for player, pixel in player_tag.items():
                if pixel is None:
                    world_coords_list.append({player: {"x": None, "y": None}})
                    continue
                pixels_raw_tensor = torch.tensor(pixel, dtype=torch.float32)
                pixels = pixels_raw_tensor.view(1, 2)
                points_image = kornia.geometry.conversions.convert_points_to_homogeneous(pixels)
                points_image = points_image.transpose(0, 1)
                h_img2img = (S @ T_adjusted @ h).squeeze(0)
                projected_points = h_img2img @ points_image
                projected_points = projected_points / (projected_points[-1] + 1e-8)
                world_coords = self.pixel_to_world_coords(projected_points, blended.size[0], blended.size[1])
                world_coords_list.append({player: {"x": world_coords.x, "y": world_coords.y}})
        return world_coords_list

    def draw_player_location(self, image_id, players_world_coords):
        df_row = self.df[self.df["image_id"] == image_id].iloc[0].to_dict()

        if "checkpoint" in image_id:
                return

        image = Image.open(self.images_path / image_id).convert("RGB")
        image = T.functional.to_tensor(image)

        cam = get_camera_from_per_sample_output(df_row, self.lens_dist)
        points_line, points_circle = df_row["points_line"], df_row["points_circle"]

        if self.lens_dist:
            image = cam.undistort_images(image.unsqueeze(0).unsqueeze(0)).squeeze()
            points_line = SNProjectiveCamera.static_undistort_points(points_line.unsqueeze(0).unsqueeze(0), cam).squeeze()
            points_circle = SNProjectiveCamera.static_undistort_points(points_circle.unsqueeze(0).unsqueeze(0), cam).squeeze()
        else:
            psi = None

        fig, ax = viz.init_figure(self.image_width, self.image_height)
        ax = viz.draw_image(ax, image)
        ax.set_xlim(0, image.shape[2])
        ax.set_ylim(image.shape[1], 0)
        ax.axis('off')
        ax = viz.draw_reprojection(ax, self.object3dcpu, cam)

        for player_tag in players_world_coords:
            for player, coords in player_tag.items():
                if any(value is None for value in coords.values()):
                    continue
                theta = np.linspace(0, 2 * np.pi, 300)
                circle_points = np.array([[coords["x"] + self.radius * np.cos(t), coords["y"] + self.radius * np.sin(t), 0] for t in theta])
                test_circle = torch.Tensor(circle_points)
                with torch.no_grad():
                    points_px = cam.project_point2pixel(test_circle, lens_distortion=False).cpu()[0, 0]
                    circle2d = Polygon(
                        points_px[:, :2],
                        closed=True,
                        color="black",
                        fill=False,
                        alpha=0.5,
                        linewidth=5,
                    )
                    ax.add_patch(circle2d)

        ax = viz.draw_selected_points(
            ax,
            self.object3dcpu,
            points_line,
            points_circle,
            kwargs_outer={
                "zorder": 1000,
                "rasterized": False,
                "s": 500,
                "alpha": 0.3,
                "facecolor": "none",
                "linewidths": 3,
            },
            kwargs_inner={
                "zorder": 1000,
                "rasterized": False,
                "s": 50,
                "marker": ".",
                "color": "k",
                "linewidths": 4.0,
            },
        )
        plt.savefig(self.output_dir / f"frames/{image_id}", dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def generate_video(self):
        frame_files = sorted((self.output_dir / "frames").glob("*.jpg"))
        frame = cv2.imread(str(frame_files[0]))
        video = cv2.VideoWriter(str(f"{self.output_dir}/video.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 25, (self.image_width, self.image_height))
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            video.write(frame)
        video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video to process")
    parser.add_argument('-v', '--video_path', type=str, required=True, help='Path to the input video file')
    args = parser.parse_args()

    video_calibration = VideoCalibration(args.video_path)
    video_calibration.extract_frames()
    video_calibration.segment_frames()
    video_calibration.calibrate_camera()
    for frame_id, tags in tqdm(video_calibration.players_tags.items(), desc="Frames iteration", total=len(video_calibration.players_tags)):
        for player in tags:
            field_coords = video_calibration.get_world_coordinates(frame_id, tags)
            video_calibration.draw_player_location(frame_id, field_coords)
    video_calibration.generate_video()
