# SAM 2 Video Masklet Tracker

This project provides: 

* A demonstration notebook that utilizes META's `SAM 2` model to track masklets in videos. The notebook allows users to select objects in a video through clicks, track the masklets throughout the video, composite the masks with the original images, and generate a final video from the masked frames.

* A script `scripts/masklet_tracker.py`, which implements the `MaskletTracker` class. This script takes player annotations and a video path as input, using the `SAM 2` model to track the players throughout the video. It optionally saves the tracked masklets in a designated folder.

## Getting Started

### Installation

#### Requirements

- Python ≥ 3.10, PyTorch ≥ 2.3.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. Install them together at https://pytorch.org to ensure this.
  * Note older versions of Python or PyTorch may also work. However, the versions above are strongly recommended to provide all features such as `torch.compile`.
- [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match the CUDA version for your PyTorch installation. This should typically be CUDA 12.1 if you follow the default installation command.

Then, install `SAM 2` from the root of this repository via
```bash
pip install -e ".[demo]"
```

#### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
cd checkpoints
bash download_ckpts.sh
```

### Demo Notebook

To use the Video Masklet Tracker demo notebook, open the notebook `video_masklet_tracker.ipynb`, located in the `notebooks/` directory. You can access it directly [here](notebooks/video_masklet_tracker.ipynb).

### MaskletTracker

The `MaskletTracker` class is designed to track player masklets in a video based on provided annotations. To instantiate this class, the following arguments are required:

- `video_path` (str | Path): The path to the video file. This is a required argument and specifies the video in which the masklets will be tracked.
- `annotations_path` (str | Path): The path to a JSON file containing player annotations. This is also required and provides the necessary data for tracking.
- `output_path` (str | Path | None): The optional path to a directory where the output masks and images will be saved. If not provided, the results will not be saved.
- `verbose` (int): This optional argument controls the level of logging. The default is `0`, which corresponds to no logging. Higher values enable more detailed logging (1 for errors, 2 for info, 3 for debug).