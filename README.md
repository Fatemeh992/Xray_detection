# Pneumonia Detection from Chest X-rays

This repository contains a project that uses **Transfer Learning** (with **MobileNetV2**) and **OpenCV**-based preprocessing to detect pneumonia in chest X-ray images. The dataset is sourced from Kaggle’s [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

---

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Requirements](#requirements)  
5. [Project Steps](#project-steps)  
6. [Results](#results)  
7. [Key Insights](#key-insights)  
8. [References](#references)  
---

## Overview

Medical imaging analysis often requires **accurate classification** to assist healthcare professionals in diagnosing illnesses. This project demonstrates how to:
- Preprocess chest X-ray images (resize, contrast enhance, normalize).
- Utilize a pre-trained CNN (MobileNetV2) via Transfer Learning.
- Fine-tune the model for optimal performance on detecting pneumonia.
- Evaluate the model using confusion matrix, accuracy, precision, recall, and specificity.

**Key Goal:** Achieve **high recall** for pneumonia (minimizing missed cases) while maintaining a reasonable specificity.

---

## Features

- **OpenCV Preprocessing**: Resize images, apply CLAHE for contrast, normalize pixel values, and convert grayscale to RGB.
- **Transfer Learning**: Start from MobileNetV2 (trained on ImageNet) and adapt it to classify **Normal** vs. **Pneumonia**.
- **Fine-Tuning**: Unfreeze top layers to further refine model performance.
- **Metrics & Visualizations**: Plot training/validation curves, confusion matrix, and compute detailed metrics (precision, recall, specificity, F1-score, etc.).

---

## Project Structure

Typical folder structure after downloading the Kaggle dataset might look like:

- **chest_xray/**: Contains the **train**, **val**, and **test** directories.  
- **pneumonia_detection.ipynb**: Main notebook with all the steps (preprocessing, model training, evaluation).  
- **requirements.txt**: Python dependencies.  
- **README.md**: This file.



---

## Requirements

- **Python 3.7+**
- **OpenCV** (for image processing)
- **TensorFlow / Keras** (for deep learning)
- **NumPy** and **Pandas**
- **Matplotlib / Seaborn** (for plotting)
- **Scikit-learn** (for metrics)
- **kagglehub** (if you are downloading directly from Kaggle in your code)

Install via:
pip install -r requirements.txt

## Project Steps

## Data Preparation
Download Dataset
Option A: Manually from Kaggle and place the extracted chest_xray folder in your project directory.
Option B: Use kagglehub within your notebook or script:

import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
Check Folder Structure and make sure you have train, val, and test folders, each containing NORMAL and PNEUMONIA subfolders.


# OpenCV Preprocessing

Read images in grayscale
Resize to (224, 224)
Apply CLAHE
Normalize pixel values
Expand to 3 channels for MobileNetV2
Load MobileNetV2

# Model Training Steps
Add Custom Layers
Compile & Train
Fine-Tuning
Unfreeze last ~30 layers for deeper training
Lower learning rate to avoid catastrophic forgetting
Evaluation & Metrics
Test Accuracy
Confusion Matrix
Classification Report (Precision, Recall, F1)
ROC Curve
Save the Model
Visualizations
Plot training vs. validation accuracy/loss curves
Display confusion matrix with seaborn.heatmap
## Results & Insights
High Recall: The model typically excels at detecting pneumonia cases (low false negatives).
Trade-Off: Specificity may drop if the threshold is set to catch more pneumonia cases.
Practical Considerations: In a clinical setting, missing pneumonia can be critical, so a high recall is often preferred over a slightly higher false positive rate.
## References
Kaggle Dataset: Chest X-Ray Images (Pneumonia)
MobileNetV2 Paper: Sandler, Mark et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks

