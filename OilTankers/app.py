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


def load_model():
    model = YoloNet(lr=2.5e-5, weight_decay=1e-4, train_dataloader=None, valid_dataloader=None)
    model.load_state_dict(torch.load(f'{__location__}/yolonet_.pt'))
    return model


def load_dataset():
    feature_extractor = AutoFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    dataset = CustomDataset(imgage_folder=(f'{__location__}/imgs'),
                            annotation_file=f'{__location__}/imgs/labels_coco.json',
                            feature_extractor=feature_extractor)
    return dataset


def plot_bboxes(test_dataset, remove_rectangles, slider_sides_diff):
    images = list()
    images_pred = list()

    random_idxs = np.random.choice(len(test_dataset), 2)
    categories = {k: v['name'] for k, v in test_dataset.coco.cats.items()}
    for random_idx in random_idxs:
        image_idx = test_dataset.coco.getImgIds()[random_idx]
        image_name = test_dataset.coco.loadImgs(int(image_idx))[0]['file_name']
        image_path = f'{__location__}/imgs/{image_name}'

        annotations = test_dataset.coco.imgToAnns[image_idx]
        image = annotate_image(image_path, annotations, categories)
        images.append(image)

        pixel_values, target = test_dataset[random_idx]
        pixel_values = pixel_values.unsqueeze(0)
        image = annotate_image_predicted(yolo_model, pixel_values, image_path, 0.0, remove_rectangles, slider_sides_diff)
        images_pred.append(image)

    st.markdown('**Annotated Bounding Boxes**')
    st.image(images, width=350, use_column_width=False)
    st.markdown('**Predicted Bounding Boxes**')
    st.image(images_pred, width=350, use_column_width=False)


st.set_page_config(page_title='Tanker Detection')
st.title('Floating Head Oil Tanker Detection')

st.markdown("""YOLO model was trained to identify floating head oil tankers.
                Click 'Load Random Images' button on the sidebar to display 2 annotated 
                images and predicted bounding boxes.""")

yolo_model = load_model()
test_dataset = load_dataset()

#Maximum Sides Difference
remove_rectangles = st.sidebar.checkbox('Remove Rectangles', value=True)
slider_sides_diff = st.sidebar.slider('Maximum Sides Difference in %', 0, 25, 10, 5) / 100

random_run = st.sidebar.button('Load Random Images')

if random_run:
    plot_bboxes(test_dataset, remove_rectangles, slider_sides_diff)



