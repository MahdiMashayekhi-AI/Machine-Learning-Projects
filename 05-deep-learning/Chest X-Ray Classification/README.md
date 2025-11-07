# 🩺 Chest X-Ray Pneumonia Classification

*A Deep Learning Project for Detecting Pneumonia from Chest X-Rays using ResNet50 (Fine-Tuning with PyTorch)*

---

## 🌍 Overview

Pneumonia is an **infection that inflames the air sacs in the lungs**, often caused by bacteria or viruses. It’s one of the **leading causes of death worldwide**, especially in children and elderly patients.
Radiologists typically identify pneumonia through **chest X-rays**, but manual diagnosis is **time-consuming and error-prone**.

This project demonstrates how **deep learning** can assist medical professionals by automatically classifying chest X-rays as:

* 🟢 **Normal**
* 🔴 **Pneumonia**

---

## 🧠 Model Summary

| Property          | Details                                     |
| ----------------- | ------------------------------------------- |
| **Base Model**    | ResNet50 (Pretrained on ImageNet)           |
| **Approach**      | Transfer Learning + Fine-tuning             |
| **Framework**     | PyTorch                                     |
| **Loss Function** | `CrossEntropyLoss`                         |
| **Optimizer**     | Adam (lr=0.0001)                             |
| **Output**        | Binary Classification (Normal vs Pneumonia) |
| **Deployment**    | Streamlit Web App                           |

---

## 📊 Dataset

* 📁 **Source:** [Kaggle – Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* 🧍‍♂️ **Classes:** `NORMAL`, `PNEUMONIA`
* 📈 **Split:**

  * Train → 80%
  * Validation → 10%
  * Test → 10%

> The dataset contains over **5,000 X-ray images** labeled by medical professionals.

---

## 🧩 Data Augmentation

Data augmentation was applied to make the model more robust against overfitting and enhance generalization:

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.9, 1.0)),
    transforms.RandomRotation(degrees=(-5, 5)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3))
    ], p=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

---

## ⚙️ Training Setup

| Parameter     | Value      |
| ------------- | ---------- |
| Epochs        | 20         |
| Batch Size    | 16         |
| Optimizer     | Adam       |
| Learning Rate | 0.0001     |
| Hardware      | GPU (CUDA) |

---

## 📈 Model Performance

| Metric              | Value                                          |
| ------------------- | ---------------------------------------------- |
| ✅ Training Accuracy | ~98%                                           |
| 🧪 Test Accuracy    | ~92%                                           |
| ⚠️ Overfitting      | Controlled using data augmentation and dropout |

---

## 💻 Model Architecture (ResNet50 Fine-Tuning)

```python
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, n_classes)
)
```

> The pretrained ResNet50 backbone was fine-tuned by replacing its final fully connected layer with a sequnetioal.

---

## 🩻 Example Predictions

| Chest X-Ray                               | Model Prediction |
| ----------------------------------------- | ---------------- |
| ![normal](assets/normal_sample.jpg)       | 🟢 **Normal**    |
| ![pneumonia](assets/pneumonia_sample.jpg) | 🔴 **Pneumonia** |

---

## 🚀 Streamlit Deployment

The model is deployed as an **interactive web app** built with **Streamlit**.
Users can upload a chest X-ray, and the app returns the model’s prediction instantly.

### Run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The interface displays:

* Uploaded image preview
* Model prediction (`Normal` / `Pneumonia`)
* Confidence score

---

## 📁 Project Structure

```
ChestXRay-Pneumonia-Classification/
│
├── data/
│   ├── train/
│   ├── test/
│   └── val/
│
├── models/
│   └── resnet50_chest.pth
│
├── app.py               # Streamlit Web Application
├── train.ipynb          # Jupyter Notebook for Training
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🧬 Future Work

* [ ] Add **Grad-CAM** visualizations for interpretability
* [ ] Integrate **OpenCV** for pre-processing & lung segmentation
* [ ] Deploy on **Streamlit Cloud / HuggingFace Spaces**
* [ ] Experiment with **EfficientNet / DenseNet**
* [ ] Add REST API using **FastAPI**

---

## 👨‍⚕️ Medical Context

**Pneumonia** appears as **white opaque regions** (lung infiltrates) on X-rays, whereas healthy lungs appear darker due to air content.
The model learns these patterns automatically, mimicking radiologist interpretation.

### Normal Lung   

<img src="https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg" alt="Normal" width="200"> 

### Pneumonia Affected

<img src="https://upload.wikimedia.org/wikipedia/commons/8/81/Chest_radiograph_in_influensa_and_H_influenzae%2C_posteroanterior%2C_annotated.jpg" alt="Pneumonia" width="200"/>

---

## 👨‍💻 Author

**Mahdi Mashayekhi**
Machine Learning Engineer

📧 Email: *[[mahdimashayekhi.ai@gmail.com](mailto:mahdimashayekhi.ai@gmail.com)]*
🌐 LinkedIn: *[MahdiMashayekhi](https://www.linkedin.com/in/mahdimashayekhi/)*
💻 GitHub: *[MahdiMashayekhi-ai](https://github.com/MahdiMashayekhi-AI)*

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It should **not be used for medical diagnosis** or patient treatment decisions.
