# SAM 2 Video Masklet Tracker

This project provides a notebook that utilizes META's SAM 2 model to track masklets in videos. The notebook allows users to select objects in a video using clicks, track the masklets through the video, composite the masks with the original images, and finally, generate a video from the masked frames.

## Getting Started

### Installation

```bash
pip install -e ".[demo]"
```
### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
cd checkpoints
bash download_ckpts.sh
```

### Video Masklet Tracker

To use the Video Masklet Tracker, open the notebook `video_masklet_tracker.ipynb`, located in the `notebooks/` directory. You can access it directly [here](notebooks/video_masklet_tracker.ipynb).
