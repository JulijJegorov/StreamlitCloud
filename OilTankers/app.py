import os
import torch
import streamlit as st
from yolonet import YoloNet

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

st.set_page_config(page_icon="🐤", page_title="Twitter Sentiment Analyzer new")

st.write('<base target="_blank">', unsafe_allow_html=True)
st.text('Hi Streamlit Cloud')
st.text(__location__)

model = YoloNet(lr=2.5e-5, weight_decay=1e-4, train_dataloader=None, valid_dataloader=None)


model.load_state_dict(torch.load(f'{__location__}/yolonet_.pt'))