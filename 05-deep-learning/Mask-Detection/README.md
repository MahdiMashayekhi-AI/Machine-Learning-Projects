# 🧠 Face Mask Detection Project  
**Mask vs No-Mask Classification using Deep Learning**

---

## 📋 Table of Contents  
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Getting Started](#getting-started)  
4. [Model Architecture & Training](#model-architecture--training)  
5. [Usage / Inference](#usage--inference)  
6. [Results](#results)  
7. [Future Work](#future-work)  
8. [License & Acknowledgements](#license--acknowledgements)

---

## 🧾 Project Overview  
This project aims to build a **Convolutional Neural Network (CNN)** that can classify whether a person in an image is wearing a **face mask** or **not**.  
The goal is to create a **practical** and **lightweight** model for real-time use cases such as surveillance, access control, or public safety systems.

### 🔍 Key Features:
- Load and preprocess facial images with/without masks  
- Train a CNN from scratch for binary classification  
- Evaluate model performance on validation data  
- Perform **real-time mask detection** using webcam feed  
- Provide clean and reproducible code for others to build upon  

---

## 📦 Dataset  
We use the **Face Mask Detection Dataset** from Kaggle:  
👉 [https://www.kaggle.com/datasets/omkargurav/face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

**Dataset details:**
- Total images: ~7,553  
- Classes:
  - `with_mask/` → 3,725 images  
  - `without_mask/` → 3,828 images  
- Each folder contains labeled RGB images suitable for binary classification tasks.

📝 **Note:**  
Download the dataset manually and extract it to your project’s `dataset/` folder.


---

## 🚀 Getting Started  

### 🧰 Prerequisites
- Python 3.7+
- PyTorch  
- torchvision  
- OpenCV  
- numpy, PIL  

### 💻 Installation  
```bash
# Activate conda environment
conda activate your_env_name

# Install dependencies
pip install torch torchvision opencv-python pillow numpy
```

### Data Structure

- Mask-Detection/
- │
- ├── app.py               # webcam & inference script
- ├── train.py             # model training script
- ├── model.py             # CNN model definition
- ├── dataset/             # dataset (from Kaggle)
- ├── README.md            # this file
- └── .gitignore

### 🧠 Model Architecture & Training

We define a simple CNN with convolutional, pooling, and fully connected layers.
The model is trained using CrossEntropyLoss and Adam optimizer over multiple epochs.

#### ⚙️ Example Training Loop

```python
  for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    validate(model, val_loader)

```

#### 📊 Experiment Summary:

| Metric   | Training | Validation |
| :------- | :------- | :--------- |
| Loss     | 0.0339   | 0.1909     |
| Accuracy | 98.39%   | 92.26%     |


These results indicate strong generalization and robust learning for a small dataset.

### 🎥 Usage / Inference

We provide app.py for real-time mask detection using your webcam.

#### ▶️ Run in Real-Time Mode

```bash
python app.py
```

- Opens webcam feed
- Mirrors the frame for natural view
- Classifies each frame (Mask / No Mask)
- Displays colored labels (✅ green for mask, ❌ red for no mask)
- Press q to quit the application.

#### 🖼️ Single Image Mode

To predict a single image instead of webcam:

```bash
python app.py --image path_to_image.jpg
```

### 📊 Results

| Metric                  | Value      |
| :---------------------- | :--------- |
| Final Training Loss     | **0.0339** |
| Final Training Accuracy | **98.39%** |
| Validation Loss         | **0.1909** |
| Validation Accuracy     | **92.26%** |

These metrics show that the CNN model performs reliably on unseen data.
Real-time webcam tests confirmed robust classification in most lighting and pose conditions.

### 🔮 Future Work

To improve and extend this project:

- ⚡ Transfer Learning: Use pretrained backbones like ResNet18 or MobileNetV2
- 🧩 Face Detection + Mask Classification: Combine with a face detector (OpenCV, SSD, YOLO)
- 💡 Add "Improper Mask" class for multi-class classification
- 🌍 Deploy Model: Convert to ONNX or TensorRT for edge devices (e.g., Raspberry Pi)
- ⚖️ Bias Analysis: Ensure fairness across different skin tones, genders, and mask types

### 📄 License & Acknowledgements

- This project is for educational purposes only.
- Dataset: Omkar Gurav – Kaggle Face Mask Dataset
- Libraries used: PyTorch, Torchvision, OpenCV, NumPy, PIL

### 💬 Author

Developed by Mahdi Mashayekhi

If you find this project useful, please ⭐ the repo and share feedback!