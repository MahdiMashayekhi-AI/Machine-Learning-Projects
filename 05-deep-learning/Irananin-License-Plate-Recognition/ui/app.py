import io
import cv2
import requests
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config("License Plate Detection (ILP)")

st.title("License Plate Detection 🚗")
st.write("Upload image of cars to detect the plates.")

API_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

  col1, col2 = st.columns(2)

  image = Image.open(uploaded_file)
  col1.image(image, caption="Original Image", use_column_width=True)

  with st.spinner("Detecting license plates..."):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()

    files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
      results = response.json().get("data", [])

      if not results:
        st.warning("No license plates detected.")
      else:
        img_array = np.array(image)

        if img_array.shape[2] == 4:
          img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)


        for result in results:
          x1, y1, x2, y2 = result["bbox"]
          label = result["text"]


          cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        col2.image(img_array, caption="Detected License Plates", use_column_width=True)

        st.write("Detected License Plates:")
        st.table(results)

    else:
      st.error("Error detecting license plates.")