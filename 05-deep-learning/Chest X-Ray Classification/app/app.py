# chest_app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------------
# 1. Page Config
# -----------------------------
st.set_page_config(
    page_title="Chest X-ray Pneumonia Detection",
    layout="centered",
    page_icon="🫁"
)

# -----------------------------
# 2. Title and Intro
# -----------------------------
st.title("🫁 Chest X-ray Pneumonia Classifier (ResNet50 + PyTorch)")
st.markdown("""
This web app uses a **fine-tuned ResNet50** model trained on the  
[Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
to detect **Pneumonia** vs **Normal** X-ray images.

Upload an image below to analyze it.
""")

# -----------------------------
# 3. Load model
# -----------------------------
@st.cache_resource
def load_model(path="../best_model_weights.pth"):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# -----------------------------
# 4. Transform (match training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 5. Prediction function
# -----------------------------
def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).numpy()[0]
    classes = ["Normal", "Pneumonia"]
    idx = np.argmax(probs)
    return classes[idx], probs

# -----------------------------
# 6. Upload and Display
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    st.markdown("⏳ **Analyzing...**")

    pred_class, probs = predict_image(image)
    confidence = probs[np.argmax(probs)]

    st.markdown("---")
    st.subheader("🔍 Prediction Results")
    st.markdown(f"**Prediction:** `{pred_class}`")
    st.progress(float(confidence))
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal", f"{probs[0]*100:.2f}%")
    with col2:
        st.metric("Pneumonia", f"{probs[1]*100:.2f}%")

    if pred_class == "Pneumonia":
        st.warning("⚠️ Possible Pneumonia detected. Please consult a healthcare professional for further evaluation.")
    else:
        st.success("✅ The X-ray appears Normal according to the model.")

# -----------------------------
# 7. Footer
# -----------------------------
st.markdown("""
---
**Model:** ResNet50 (Fine-tuned)  
**Framework:** PyTorch + Streamlit  
**Author:** Mahdi Mashayekhi  
**Dataset:** Kaggle Chest X-ray Pneumonia  
""")
