# 🐶🐱 Cat vs Dog Image Classifier

This project implements a **Convolutional Neural Network (CNN)** in **PyTorch** to classify images of cats and dogs.  
The model is trained on a subset of images from Kaggle and demonstrates image classification using deep learning techniques.

---

## 📂 Dataset

The dataset used in this project can be downloaded from Kaggle:

[Cat and Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)

### Dataset Structure
The dataset should be organized as follows:

- dataset/
- ├── training_set/
- │ ├── cats/
- │ └── dogs/
- └── test_set/
- ├── cats/
- └── dogs/


Each folder contains the corresponding images for training and testing.

---

## 🧠 Project Overview

**Goal:** Build a CNN that can accurately distinguish between cat and dog images.  

**Key Steps:**
1. **Data Loading & Preprocessing:**  
   - Resize images to a consistent size (e.g., 64x64).  
   - Normalize pixel values.  
   - Apply **data augmentation** (random flips, rotations, color jitter) to improve generalization.  

2. **Model Architecture:**  
   - Multiple convolutional layers with ReLU activation and MaxPooling.  
   - Batch Normalization to stabilize training.  
   - Dropout layers to reduce overfitting.  
   - Fully connected layers ending with 2 outputs (cat vs dog).  

3. **Training & Evaluation:**  
   - Use **CrossEntropyLoss** as the criterion.  
   - Optimizer: **Adam** with weight decay.  
   - Optional: **Learning rate scheduler** (e.g., OneCycleLR) for faster convergence.  
   - Track training and validation accuracy and loss per epoch.  

4. **Prediction:**  
   - Functions to predict single images or images from URLs.  
   - Model outputs either "Cat" or "Dog" based on highest probability.

---

## 📈 Results

After training, the model can:
- Achieve high accuracy on validation images.  
- Predict unseen images reliably with proper preprocessing.  
- Serve as a base for **real-time applications** such as webcam-based detection.

---

## ⚙️ Usage

1. Clone the repository:
```bash
git clone <your-repo-url>
cd cat-dog-classifier
```

2. Download and organize the dataset as described above.

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 💡 Notes & Tips

- Data Augmentation: Helps reduce overfitting and improves generalization.
- Normalization: Crucial for CNN performance.
- Model Checkpoints: Save weights after each epoch for safety.
- Evaluation: Always check accuracy on a separate test set to avoid data leakage.

---

## 📝 References

- PyTorch official documentation: https://pytorch.org/docs/stable/index.html

- Kaggle Dataset: [Cat and Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)