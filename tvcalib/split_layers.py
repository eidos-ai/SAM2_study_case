import cv2
import numpy as np
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt

image_path = 'data/soccernet/1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
image_tensor = transform(image_rgb)

with torch.no_grad():
    prediction = model([image_tensor])

masks = prediction[0]['masks'].numpy()
scores = prediction[0]['scores'].numpy()
threshold = 0.5

players_layer = np.zeros_like(image_rgb)
background_layer = image_rgb.copy()

for i in range(len(masks)):
    if scores[i] > threshold:
        mask = masks[i, 0]
        players_layer[mask > threshold] = image_rgb[mask > threshold]
        background_layer[mask > threshold] = 0

players_layer_bgr = cv2.cvtColor(players_layer, cv2.COLOR_RGB2BGR)
background_layer_bgr = cv2.cvtColor(background_layer, cv2.COLOR_RGB2BGR)
combined_image_bgr = background_layer_bgr.copy()
mask = np.any(players_layer_bgr != 0, axis=-1)
cv2.circle(combined_image_bgr, center=(877, 650), radius=25, color=(255,0,0), thickness=3)
combined_image_bgr[mask] = players_layer_bgr[mask]
cv2.imwrite('tmp/combined_layers.jpg', combined_image_bgr)
