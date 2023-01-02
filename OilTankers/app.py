import os
import torch
import streamlit as st

from yolonet import YoloNet
from custom_dataset import CustomDataset
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoFeatureExtractor

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

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



st.set_page_config(page_icon="🐤", page_title="Twitter Sentiment Analyzer new")

st.write('<base target="_blank">', unsafe_allow_html=True)
st.text('Hi Streamlit Cloud')
st.text(__location__)

feature_extractor = AutoFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
dataset = CustomDataset(imgage_folder=(f'{__location__}/imgs'),
                        annotation_file=f'{__location__}/imgs/labels_coco.json',
                        feature_extractor=feature_extractor)

st.text(dataset)

model = YoloNet(lr=2.5e-5, weight_decay=1e-4, train_dataloader=None, valid_dataloader=None)




model.load_state_dict(torch.load(f'{__location__}/yolonet_.pt'))

st.text(model)