# Diabetes Prediction App

## Overview

This project involves building a machine learning model to predict whether an individual is diabetic or non-diabetic based on images. The model utilizes a pre-trained VGG16 architecture, fine-tuned for this specific classification task. A Streamlit application provides a user-friendly interface for uploading images and receiving predictions.

## Project Structure

1. **Data Preparation**: Images are loaded, preprocessed, and labeled from the specified directories. The dataset is then split into training and testing sets.
2. **Model Training**: The VGG16 model, pre-trained on ImageNet, is adapted with custom dense layers to classify images into diabetic or non-diabetic categories. The model is then fine-tuned and saved.
3. **Model Evaluation**: The performance of the trained model is evaluated on a test dataset to determine its accuracy.
4. **Streamlit App**: The app allows users to upload images or choose from sample images to get predictions about diabetes status.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Streamlit
- Pillow

You can install the necessary packages using pip:

```bash
pip install tensorflow numpy opencv-python streamlit pillow
```

## Dataset

The dataset is organized into:

- **Train**:
  - `Control Group` (Non-Diabetic)
  - `DM Group` (Diabetic)
- **Validation**:
  - `Control Group` (Non-Diabetic)
  - `DM Group` (Diabetic)

Ensure images are resized to 224x224 pixels.

## Model Training

The VGG16 model is fine-tuned for the diabetes prediction task. For details on training, including dataset preparation and model configuration, please refer to the code in the GitHub repository.

## Model Evaluation

The model's performance is assessed on the test dataset to verify its accuracy. For specifics on evaluation metrics, see the code provided on GitHub.

## Streamlit Application

The Streamlit app enables users to:

1. Upload an image or select from a set of sample images.
2. Receive predictions on whether the individual is diabetic or non-diabetic.

To run the Streamlit app, follow the instructions provided in the GitHub repository.

For more information and to download the code, please visit the [GitHub repository](https://github.com/MuhammadArslan007/diabetes-prediction-app.git).

