"""
    Author: julij
    Date: 02/01/2023
    Description: 
"""

import torch
from PIL import Image, ImageDraw, ImageFont

COLORS = ['#1F497D', '#4F81BD',  '#C0504D', '#9BBB59', '#8064A2', '#4BACC6', '#F79646', '#6B7C87', '#8064A2', '#00728C', '#6a1635', '#8EC3D8']
COLORS_CYCLE = COLORS * 150

def annotate_image(image_path: str, annotations: dict, categories: dict):
  image = Image.open(image_path)
  image_draw = ImageDraw.Draw(image, 'RGBA')

  for annotation in annotations:
    box = annotation['bbox']
    class_idx = annotation['category_id']
    x, y, w, h = tuple(box)
    image_draw.rectangle((x, y, x+w, y+h), outline='red', width=1)
    image_draw.text((x, y), categories[class_idx], fill='white')
  return image


def annotate_image_predicted(model, pixel_values, image_path, threshold):
  image = Image.open(image_path)
  model.eval()
  with torch.no_grad():
    outputs = model(pixel_values=pixel_values)

  probas = outputs.logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold
  bboxes = outputs.pred_boxes[0, keep]
  bboxes_scaled = rescale_bboxes(bboxes, image.size)

  image_draw = ImageDraw.Draw(image, 'RGBA')
  for idx, (xmin, ymin, xmax, ymax) in enumerate(bboxes_scaled.tolist()):
    image_draw.rectangle((xmin, ymin, xmax, ymax), outline=COLORS_CYCLE[idx], width=2)

  return image



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
  img_w, img_h = size
  b = box_cxcywh_to_xyxy(out_bbox)
  b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
  return b


