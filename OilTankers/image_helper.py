"""
    Author: julij
    Date: 02/01/2023
    Description: 
"""

import torch
from PIL import Image, ImageDraw, ImageFont


def annotate_image(image_path: str, annotations: dict, categories: dict ):
  image = Image.open(image_path)
  image_draw = ImageDraw.Draw(image, 'RGBA')

  for annotation in annotations:
    box = annotation['bbox']
    class_idx = annotation['category_id']
    x, y, w, h = tuple(box)
    image_draw.rectangle((x, y, x+w, y+h), outline='red', width=1)
    image_draw.text((x, y), categories[class_idx], fill='white')
  return image


