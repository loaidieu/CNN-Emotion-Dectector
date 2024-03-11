# CNN Emotion Detector

## Project Introduction

Welcome to CNN Emotion Detector, where the future of AI-driven feedback takes shape. This project aims to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch to analyze facial images and categorize facial expression into distinct states: Neutral, Engaged/Focused, Surprised, and Happy.

[Our GitHub Repo](https://github.com/loaidieu/CNN-Emotion-Dectector)

- :neutral_face: **Neutral**
- :smiley: **Happy**
- :astonished: **Surprised**
- :nerd_face: **Engaged**

## Contents

### Python Scripts

- `data_processing.py`: Manages the cleaning/preprocessing of the data.
- `data_visualization.py`: Includes code to generate visualizations of the dataset.
- `main.py`: Serves as the main driver of our data cleaning/processing and visualization tasks.

### Dataset

A document that lists the provenance of each dataset/image used in this project. It includes details such as dataset name, source, and licensing type. The document also links to the full dataset repository.

### Representative Images

Includes 25 representative images for each class: Neutral, Engaged/Focused, Surprised, and Happy.

### `README.md`

This file. Provides an overview of the project, describes the purpose of each file, and outlines the steps to execute the project's code for data cleaning and visualization.

## Execution/Setup Instructions

First, you need to have Python and `pip`, the Python package manager, installed on your system.

### Installing Python and `pip`

1. **Python Installation**:
   - If you don't already have Python installed, download it from [the official Python website](https://www.python.org/downloads/). Installing Python will also install `pip`.

2. **Verifying Installation**:
   - After installation, you can verify that Python and `pip` are installed by opening a terminal (Command Prompt on Windows, Terminal on macOS and Linux) and typing:
     ```bash
     python --version
     pip --version
     ```
   If both commands return version numbers, Python and `pip`. Congratz it means you have it install on your machine.

3. **Install Required Libraries**:
   - To excute the script we need to install some libraries Open your terminal or command prompt and run the following command:
     ```bash
     pip install torch torchvision numpy matplotlib scikit-learn pillow
     ```
   This command will install the required libraries to run the code.

4. **Running the Scripts**:
   - You can to run the script with the following command in your terminal:
     ```bash
     python main.py
     ```
   This will trigger the data cleaning, processing, and visualization of the dataset.
