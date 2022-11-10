import streamlit as st
import requests
from PIL import Image
import numpy as np
from requests_toolbelt.multipart.encoder import MultipartEncoder

# interact with FastAPI endpoint
PREDICT_URL = "http://127.0.0.1:8000/predict"

def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image")})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r

st.write("""
    # Face Shape Estimator
""")
uploaded_file = st.file_uploader("Upload Face Image!!", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    col1, col2= st.columns(2)     

    image = Image.open(uploaded_file)
    with col1:
        st.header("Input Face Image")
        st.image(image)

    with col2:
        st.header("Result")
        res = process(uploaded_file, PREDICT_URL)
        res = res.json()['label']
        st.write(f'Your face shape is "{res}"')