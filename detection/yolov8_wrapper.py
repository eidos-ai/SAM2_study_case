import os
import json
from typing import List
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import torch

class YoloV8Wrapper:
    def __init__(self, model_weights: Path, 
                 conf: float=0.25, nms_iou: float=0.25, half: bool=False, imgsz: int=640,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Load the model
        self.model = YOLO(model_weights).to(device)
        self.conf = conf
        self.nms_iou = nms_iou
        self.half = half
        self.imgsz = imgsz
        self.device = device

    def predict_on_img_list(self, image_paths: List[str]):
        # Make predictions and format the results
        formatted_results = []
        for img_path in tqdm(image_paths, desc="Inference on list of images"):
            detections = []
            img_results = self.model.predict(img_path, verbose=False, 
                                            conf=self.conf, iou=self.nms_iou, half=self.half,
                                            imgsz=self.imgsz)[0]
            
            for bbox, pred_class, pred_prob in zip(img_results.boxes.xywhn.tolist(), img_results.boxes.cls.tolist(), img_results.boxes.conf.tolist()):
                cx, cy, w, h = bbox
                detections.append({
                    "bbox": [cx, cy, w, h],
                    "class": int(pred_class),
                    "score": pred_prob
                })
            formatted_results.append({
                "image_path": img_path,
                "detections": detections
            })

        return formatted_results
    
    def predict(self, model_input, save: bool=False):
        return self.model.predict(model_input, save=save, verbose=False,
                                  conf=self.conf, iou=self.nms_iou, half=self.half,
                                  imgsz=self.imgsz)
        
def save_list_of_dicts_to_json(data, file_path):
    """
    Save a list of dictionaries to a JSON file.

    Args:
    - data (list): A list of dictionaries to be saved to JSON.
    - file_path (str): Path of the file where data will be saved.
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True, help="Path to trained yolo model")
    parser.add_argument("-i", "--input", required=True, help="Path to image or directory with images")
    parser.add_argument("--save", required=False, action="store_true", default=False, help="save predicted images and video")
    args = parser.parse_args()
    
    model = YoloV8Wrapper(args.model_path)
    
    results = model.predict(args.input, args.save)
    
    if args.save:
        output_detections = [{"image_path": r.path, 
                            "detections": [{"bbox": box.xywhn.tolist(), 
                                            "class": int(box.cls.item())} for box in r.boxes]} for r in results]
        save_list_of_dicts_to_json(output_detections, os.path.join(results[0].save_dir, "detections.json"))