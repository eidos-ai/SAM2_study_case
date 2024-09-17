import random
import json
import cv2
import shutil
import kornia
import torch
import matplotlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Circle
from collections import defaultdict
from argparse import Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from SoccerNet.Evaluation.utils_calibration import SoccerPitch
from tvcalib.cam_modules import SNProjectiveCamera
from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import generate_class_synthesis, get_line_extremities
from tvcalib.sncalib_dataset import custom_list_collate, split_circle_central
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from tvcalib.inference import InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel
from tvcalib.inference import get_camera_from_per_sample_output
from tvcalib.utils import visualization_mpl_min as viz


def pixel_to_field_coords(pixel_coords, img_width, img_height):
    x_scale = 105 / img_width
    y_scale = 68 / img_height
    x_meters = pixel_coords[0] * x_scale
    y_meters = pixel_coords[1] * y_scale
    x_field = (x_meters - (105 / 2)).numpy()
    y_field = (y_meters - (68 / 2)).numpy()
    return [round(x_field[0], 2), round(y_field[0], 2)]

def process_video(video_path):
    args = Namespace(
        video_path=Path(video_path),
        images_path=Path("tmp/raw_frames"),
        output_dir=Path("tmp"),
        checkpoint="data/segment_localization/train_59.pt",
        gpu=True,
        nworkers=8,
        batch_size_seg=8,
        batch_size_calib=256,
        image_width=1920,
        image_height=1080,
        optim_steps=2000,
        lens_dist=False,
        write_masks=False
    )

    args.images_path.mkdir(parents=True, exist_ok=True)
    output_frames_path = args.output_dir / "frames"
    output_frames_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video_path))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = args.images_path / f"{frame_count + 1:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_count += 1
    cap.release()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

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

    dataset_seg = InferenceDatasetSegmentation(
        args.images_path, args.image_width, args.image_height
    )
    dataloader_seg = torch.utils.data.DataLoader(
        dataset_seg,
        batch_size=args.batch_size_seg,
        num_workers=args.nworkers,
        shuffle=False,
        collate_fn=custom_list_collate,
    )

    model_seg = InferenceSegmentationModel(args.checkpoint, device)

    image_ids = []
    keypoints_raw = []
    (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    for batch_dict in tqdm(dataloader_seg):
        with torch.no_grad():
            sem_lines = model_seg.inference(batch_dict["image"].to(device))
        sem_lines = sem_lines.cpu().numpy().astype(np.uint8)

        with Pool(args.nworkers) as p:
            skeletons_batch = p.map(fn_generate_class_synthesis, sem_lines)
            keypoints_raw_batch = p.map(fn_get_line_extremities, skeletons_batch)

        if args.write_masks:
            for image_id, mask in zip(batch_dict["image_id"], sem_lines):
                mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
                mask.putpalette(lines_palette)
                mask.convert("RGB").save(args.output_dir / "masks" / image_id)

        image_ids.extend(batch_dict["image_id"])
        keypoints_raw.extend(keypoints_raw_batch)

    model_calib = TVCalibModule(
        object3d,
        get_cam_distr(1.96, args.batch_size_calib, 1),
        get_dist_distr(args.batch_size_calib, 1) if args.lens_dist else None,
        (args.image_height, args.image_width),
        args.optim_steps,
        device,
        log_per_step=False,
        tqdm_kwqargs=None,
    )

    dataset_calib = InferenceDatasetCalibration(keypoints_raw, args.image_width, args.image_height, object3d)
    dataloader_calib = torch.utils.data.DataLoader(dataset_calib, args.batch_size_calib, collate_fn=custom_list_collate)

    per_sample_output = defaultdict(list)
    per_sample_output["image_id"] = [[x] for x in image_ids]
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

    # export df as json
    dict_per_frame = [{key: value[0] for key, value in zip(per_sample_output.keys(), values)} for values in zip(*per_sample_output.values())]
    for image_dict in dict_per_frame:
        keys_to_delete = [key for key, value in image_dict.items() if isinstance(value, torch.Tensor)]
        for key in keys_to_delete:
            del image_dict[key]
    output_dict = {dic["image_id"]: {k: v for k, v in dic.items() if k != "image_id"} for dic in dict_per_frame}
    with open('tmp/metadata.json', 'w') as json_file:
        json.dump(output_dict, json_file, indent=4, sort_keys=True)

    progress_bar = tqdm(df.iterrows(), total=df.shape[0])
    for index, sample in progress_bar:
        image_id = Path(sample.image_id).stem
        progress_bar.set_description(f"Frame {image_id}")
        if "checkpoint" in image_id:
            continue

        # now filtering to specific frame for dev reasons
        # if "0001" not in image_id:
        #     continue

        image = Image.open(args.images_path / sample.image_id).convert("RGB")
        image = T.functional.to_tensor(image)
        h = torch.tensor(sample["homography"])

        target_scale = 5
        S = torch.eye(3).unsqueeze(0)
        S[:, 0, 0] = S[:, 1, 1] = target_scale
        # translate center of the homography matrix to the correct image origin (upper left)
        T_adjusted = torch.eye(3).unsqueeze(0)
        T_adjusted[:, 0, -1] = 105 / 2
        T_adjusted[:, 1, -1] = 68 / 2
        
        def _warp_img(image, H):
            warped_top = kornia.geometry.transform.homography_warp(
                image.unsqueeze(0),
                S @ T_adjusted @ H,
                dsize=(int(68 * target_scale), int(105 * target_scale)),
                normalized_homography=False,
                normalized_coordinates=False,
                mode="nearest",
            ).squeeze()
            return warped_top

        mask = _warp_img(image, h)
        mask = torch.permute(mask, (1, 2, 0))
        mask = (mask * 255).to(torch.uint8)
        mask = 255 - mask

        # overlay with template
        ftemplate = Path("template_pitch_t.png") 
        template = T.functional.to_tensor(Image.open(ftemplate).convert("RGB"))
        template = T.functional.resize(template, (int(target_scale*68), int(target_scale*105)))
        template = torch.permute(template, (1, 2, 0))
        template = (template * 255).to(torch.uint8)
        background = Image.fromarray(template.numpy())
        foreground = Image.fromarray(mask.numpy())
        blended = Image.blend(background, foreground, alpha=0.5)
        # blended.save("tmp/blended_image.jpg", "JPEG")

        # example pixel of field (0,0) in frame
        pixel = [841, 697]
        pixels_raw_tensor = torch.tensor(pixel, dtype=torch.float32)
        pixels = pixels_raw_tensor.view(1, 2)
        points_image = kornia.geometry.conversions.convert_points_to_homogeneous(pixels)
        points_image = points_image.transpose(0, 1)
        h_img2img = (S @ T_adjusted @ h).squeeze(0)
        projected_points = h_img2img @ points_image
        projected_points = projected_points / (projected_points[-1] + 1e-8)
        world_coords = pixel_to_field_coords(projected_points, blended.size[0], blended.size[1])
        print("world coords:\n", world_coords)

        # plot blended image
        # plt.imshow(blended)
        # plt.scatter(projected_points[0], projected_points[1])
        # delta_m = 25
        # plt.xlim(0-delta_m, 105*target_scale+delta_m)
        # plt.ylim(0-delta_m, 68*target_scale+delta_m)
        # plt.gca().invert_yaxis() # 'image-style' coordinate system (upper left origin)
        # plt.savefig('imagen_con_puntos.png', bbox_inches='tight', pad_inches=0)

        cam = get_camera_from_per_sample_output(sample, args.lens_dist)
        points_line, points_circle = sample["points_line"], sample["points_circle"]

        if args.lens_dist:
            image = cam.undistort_images(image.unsqueeze(0).unsqueeze(0)).squeeze()
            points_line = SNProjectiveCamera.static_undistort_points(points_line.unsqueeze(0).unsqueeze(0), cam).squeeze()
            points_circle = SNProjectiveCamera.static_undistort_points(points_circle.unsqueeze(0).unsqueeze(0), cam).squeeze()
        else:
            psi = None

        fig, ax = viz.init_figure(args.image_width, args.image_height)
        ax = viz.draw_image(ax, image)
        ax.set_xlim(0, image.shape[2])
        ax.set_ylim(image.shape[1], 0)
        ax.axis('off')
        ax = viz.draw_reprojection(ax, object3dcpu, cam)
        
        # draw example diagonal line and circle
        points_diagonal = [[[-52.5, 34, 0], [52.5, -34, 0]]]
        diagonal_line = torch.Tensor(points_diagonal)
        with torch.no_grad():
            points_px = cam.project_point2pixel(diagonal_line, lens_distortion=False).cpu()[0, 0]
            line2d = Line2D(
                points_px[:, 0],
                points_px[:, 1],
                color="red",
                alpha=0.5,
                linewidth=5,
            )
            ax.add_line(line2d)
        # draw example circle with world_coords of field (0,0) in frame 0001 as its center
        # point_center_test_circle = [10, -10]
        theta = np.linspace(0, 2 * np.pi, 300)
        radius = 1
        circle_points = np.array([[world_coords[0] + radius * np.cos(t), world_coords[1] + radius * np.sin(t), 0] for t in theta])
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
            object3dcpu,
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
        plt.savefig(args.output_dir / f"frames/{image_id}.png", dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # join frames into video
    video_path = args.output_dir / "video.mp4"
    frame_files = sorted(output_frames_path.glob("*.png"))
    frame = cv2.imread(str(frame_files[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        video.write(frame)
    video.release()
    shutil.rmtree(args.images_path)
    shutil.rmtree(output_frames_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video to process")
    parser.add_argument('-v', '--video_path', type=str, required=True, help='Path to the input video file')
    args = parser.parse_args()
    process_video(args.video_path)
