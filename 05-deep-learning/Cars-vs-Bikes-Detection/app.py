import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import requests
import io
import pandas as pd

# --- Page config ---
st.set_page_config(page_title="Car vs Motorcycle Classifier", layout="wide")

# --- Device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_names = ["motorcycle", "car"]  # Adjust based on your dataset

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Helper function ---
def predict_image(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return class_names[pred], probs[0][pred].item(), probs

# --- App UI ---
st.title("🚗🛵 Car vs Motorcycle Classifier")
st.write("Upload an image or enter a URL to see the prediction in real-time.")

# Upload file or URL
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
url_input = st.text_input("Or enter image URL:")

img = None
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
elif url_input:
    try:
        response = requests.get(url_input)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    except:
        st.error("Could not load image from URL.")

if img is not None:
    col1, col2 = st.columns([2,1])
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
    with col2:
        label, prob, probs = predict_image(img)
        
        # Prediction box
        st.markdown(
            f"""
            <div style='background-color:#f0f0f5;padding:15px;border-radius:10px'>
            <h2 style='color:#4B0082'>Prediction: {label}</h2>
            <h3>Probability: {prob*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True
        )
        
        # Probability chart
        prob_df = pd.DataFrame({
            'Class': class_names,
            'Probability': [p.item()*100 for p in probs[0]]
        })
        st.bar_chart(prob_df.set_index('Class'))
        
        # Optional: additional info
        st.write("Model Architecture: MobileNetV2 pretrained + fine-tuned classifier head")
        st.write("Input image size: 224x224, Normalized with ImageNet stats")
