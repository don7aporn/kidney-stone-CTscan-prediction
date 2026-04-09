
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("Kidney Stone Detection")

# โหลดโมเดล
model = YOLO("yolov8m.pt") # เปลี่ยนเป็น yolov8m.pt

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)

    for r in results:
        st.image(r.plot(), caption="Prediction")
