import os
print(os.listdir())

from ultralytics import YOLO
import cv2

cv2.setNumThreads(0)

import streamlit as st
from PIL import Image

import numpy as np

st.title("Kidney Stone Detection")

model = YOLO("kidney_stone_model.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # แปลง PIL เป็น numpy array
    image_array = np.array(image)

    # ทำการทำนาย
    results = model(image_array)

    # แสดงผล
    st.image(results[0].plot(), caption="Prediction", use_column_width=True)
