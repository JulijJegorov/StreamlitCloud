import os
import streamlit as st

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

st.set_page_config(page_icon="🐤", page_title="Twitter Sentiment Analyzer new")

st.write('<base target="_blank">', unsafe_allow_html=True)
st.text('Hi Streamlit Cloud')