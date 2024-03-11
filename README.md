# CNN Emotion Detector

## Project Introduction

Welcome to CNN Emotion Detector, where the future of AI-driven feedback takes shape. This project aims to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch to analyze facial images and categorize facial expression into distinct states: Neutral, Engaged/Focused, Surprised, and Happy.

- :neutral_face: **Neutral**
- :smiley: **Happy**
- :astonished: **Surprised**
- :nerd_face: **Engaged**

## Contents

### Python Scripts

- `data_cleaning.py`: Contains functions and routines to standardize the dataset by resizing images, applying light processing such as rotations, brightness adjustments, and minor cropping. 
- `data_visualization.py`: Includes code to generate visualizations of the dataset using scikit-learn and Matplotlib. It plots class distribution, sample images, and pixel intensity distributions.
- `dataset_processing.py`: Manages the preprocessing of data.
### Dataset

A document that lists the provenance of each dataset/image used in this project. It includes details such as dataset name, source, and licensing type. The document also links to the full dataset repository.

### Representative Images

Includes 25 representative images for each class: Neutral, Engaged/Focused, Surprised, and Happy.

### `README.md`

This file. Provides an overview of the project, describes the purpose of each file, and outlines the steps to execute the project's code for data cleaning and visualization.

## Execution Instructions

### Data Cleaning

To clean and prepare your dataset for training, follow these steps:

1. **Ensure Dependencies Are Installed**: Make sure you have Python, PyTorch, scikit-learn, and Matplotlib installed in your environment.
2. **Execute Data Cleaning Script**: Run `python data_cleaning.py`. This script automatically standardizes the size of the images and applies preprocessing techniques to enhance the dataset's quality.

### Data Visualization

To visualize the dataset for exploratory analysis:

1. **Run Data Visualization Script**: Execute `python data_visualization.py`. This script generates several plots to help understand the dataset's distribution and characteristics.
2. **Review Generated Plots**: Check the generated plots for class distribution, sample images per class, and pixel intensity distributions.
