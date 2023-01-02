import os
import numpy as np
import torch
import streamlit as st

from yolonet import YoloNet
from custom_dataset import CustomDataset
from PIL import Image, ImageDraw, ImageFont

from image_helper import annotate_image, annotate_image_predicted
from transformers import AutoFeatureExtractor

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


st.set_page_config(page_icon="🐤", page_title="Twitter Sentiment Analyzer new")


model = YoloNet(lr=2.5e-5, weight_decay=1e-4, train_dataloader=None, valid_dataloader=None)
model.load_state_dict(torch.load(f'{__location__}/yolonet_.pt'))



st.write('<base target="_blank">', unsafe_allow_html=True)
st.text('Hi Streamlit Cloud')
st.text(__location__)

feature_extractor = AutoFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
dataset = CustomDataset(imgage_folder=(f'{__location__}/imgs'),
                        annotation_file=f'{__location__}/imgs/labels_coco.json',
                        feature_extractor=feature_extractor)


st.text(dataset)
categories = {k: v['name'] for k, v in dataset.coco.cats.items()}



image_idxs = np.random.choice(dataset.coco.getImgIds(), 2)

random_idxs = np.random.choice(len(dataset), 2)

images = list()
images_pred = list()
for random_idx in random_idxs:
    image_idx = dataset.coco.getImgIds()[random_idx]
    image_name = dataset.coco.loadImgs(int(image_idx))[0]['file_name']
    image_path = f'{__location__}/imgs/{image_name}'

    annotations = dataset.coco.imgToAnns[image_idx]
    image = annotate_image(image_path, annotations, categories)
    images.append(image)

    pixel_values, target = dataset[random_idx]
    image = annotate_image_predicted(model, pixel_values, image_path, 0.000000000000005)
    images_pred.append(image)



st.image(images, width=300, use_column_width=False)

st.image(images_pred, width=300, use_column_width=False)




# image_name = dataset.coco.loadImgs(int(image_idx))[0]['file_name']
#
#
# annotations = dataset.coco.imgToAnns[image_idx]
# image = annotate_image(image_path, annotations, categories)




#
#
# pixel_values, target = dataset[0]
# pixel_values = pixel_values.unsqueeze(0).to('cpu')
# model.eval()
# with torch.no_grad():
#   outputs = model(pixel_values=pixel_values)
#   bboxes = outputs.pred_boxes[0].cpu()
#
# bboxes_scaled = rescale_bboxes(bboxes, image.size)
#
#
# st.text(bboxes_scaled)
#
# image_idx = dataset.coco.getImgIds()[0]
#
# image_name = dataset.coco.loadImgs(int(image_idx))[0]['file_name']
# image_path = f'{__location__}/imgs/{image_name}'
# image = Image.open(image_path)
#
# st.image(image, width=300, use_column_width=False)
