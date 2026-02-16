# Satellite-Land-Cover-Classification using Machine Learning & CNN

This project focuses on **Land Use / Land Cover (LULC) Classification** using satellite imagery.
It uses both **classical Machine Learning** and **Deep Learning (CNN)** approaches to classify
satellite images into different land cover categories such as Forest, River, Residential,
Agriculture, Highway, etc.

The project is implemented in **Python** using **Google Colab** and is suitable for beginners
who want to understand how satellite images are processed and classified.

---

## 📂 Dataset

- **Dataset Name:** EuroSAT
- **Source:** Kaggle
- **Type:** Satellite imagery (RGB images)
- **Classes:**  
  - AnnualCrop  
  - Forest  
  - HerbaceousVegetation  
  - Highway  
  - Industrial  
  - Pasture  
  - PermanentCrop  
  - Residential  
  - River  
  - SeaLake  

Each class contains satellite images representing a specific land cover type.

---

## 🧠 Project Workflow

1. Download dataset directly into Google Colab using Kaggle API  
2. Load and preprocess satellite images  
3. Convert images into numerical format  
4. Split data into training and testing sets  
5. Train models:
   - Random Forest Classifier
   - Convolutional Neural Network (CNN)
6. Evaluate model performance
7. Visualize predictions

---

## 🛠 Technologies Used

- Python
- NumPy
- Matplotlib
- Pillow (PIL)
- Scikit-learn
- TensorFlow / Keras
- Google Colab

---

## 📌 Machine Learning Approach

### Random Forest Classifier
- Images resized to 64×64
- Flattened into 1D feature vectors
- Supervised learning with labeled data
- Used as a baseline model

**Advantage:** Simple and fast  
**Limitation:** Loses spatial information

---

## 🧠 Deep Learning Approach (CNN)

### Convolutional Neural Network
- Images kept in 3D format (64×64×3)
- Automatic feature extraction using convolution layers
- Max pooling to reduce dimensionality
- Softmax output layer for classification

**Advantage:** Preserves spatial features and improves accuracy

---

## 📊 Model Performance

- Random Forest Accuracy: ~85–90%
- CNN Accuracy: ~90–95% (depending on epochs)

Evaluation metrics:
- Accuracy score
- Confusion matrix
- Visual inspection of predictions

---

## 📁 Project Structure

