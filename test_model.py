from SegFormer import SegformerForSemanticSegmentation
from transformers import SegformerConfig, SegformerImageProcessor

from ade20k_utils import ade_palette

import torch
import numpy as np
from PIL import Image
import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
metric = evaluate.load('mean_iou')

config = SegformerConfig()
processor = SegformerImageProcessor(do_resize=False)
model = SegformerForSemanticSegmentation(config=config).from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
model.load_state_dict(torch.load('saved_weights/b4_to_b0_distill_b8t3.pth', weights_only=True))

model.to(device)
model.eval()

image_path = 'test.png'
image = Image.open(image_path)

pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

with torch.no_grad():
    outputs = model(pixel_values)
    logits = outputs.logits

predicted_segmentation_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

color_seg = np.zeros((predicted_segmentation_map.shape[0],
                      predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

palette = np.array(ade_palette())
for label, color in enumerate(palette):
    color_seg[predicted_segmentation_map == label, :] = color
# Convert to BGR
color_seg = color_seg[..., ::-1]

# Show image + mask
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

img = Image.fromarray(img)
img.save("output.jpeg")