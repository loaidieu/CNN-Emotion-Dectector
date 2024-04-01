# CNN Emotion Detector

## Project Introduction

Welcome to CNN Emotion Detector, where the future of AI-driven feedback takes shape. This project aims to develop a Deep Learning Convolutional Neural Network (CNN) using PyTorch to analyze facial images and categorize facial expression into distinct states: Neutral, Engaged/Focused, Surprised, and Happy.

[Our GitHub Repo](https://github.com/loaidieu/CNN-Emotion-Dectector)

- :neutral_face: **Neutral**
- :smiley: **Happy**
- :astonished: **Surprised**
- :nerd_face: **Engaged**

## Table of Contents


## Installation

1. **Python Installation**
   - If you don't already have Python installed, download it from [the official Python website](https://www.python.org/downloads/). Installing Python will also install `pip`.

2. **Verifying Installation**
   - After installation, you can verify that Python and `pip` are installed by opening a terminal (Command Prompt on Windows, Terminal on macOS and Linux) and typing:
     ```bash
     python --version
     pip --version
     ```
   If both commands return version numbers, Python and `pip`. Congratz it means you have it install on your machine.

3. **Package Installation**
   - All the libraries required to run this project can be found in `requirements.txt`, whose installations are taken care of by `main.py`.


## Folder Organization
- Images used for train, and test ------------------------------------------------------------------> data :file_folder:
- Plot outputs (data visualizations) ---------------------------------------------------------------> data_visuals :file_folder:
- Model definitions and trained model pickles ------------------------------------------------------> models :file_folder:
- Report (along with signed Originality Forms) and original data provenance information ------------> report :file_folder:
- Confusion matrices and metrics table -------------------------------------------------------------> results :file_folder:


## Python Scripts
- `custom_dataset`: Given X and y, returns Dataset type object (for dataloader)
- `data_processing.py`: Manages the cleaning/preprocessing of the data
- `data_visualization.py`: Includes code to generate visualizations of the dataset
- `main.py`: Serves as the main driver of all other scripts
- `predict.py`: Given a dataloader object and a trained model, returns a list of predictions
- `test.py`: Given a dataloader object, a model and a loss function, returns loss and accuracy of predictions over 1 epoch
- `train.py`: Given a dataloader object, a model, a loss function and an optimizer, returns loss and accuracy of predictions over 1 epoch
- `train_loop.py`: Given an X number of epochs, runs `train.py` and `test.py` X number of times and returns train and test losses/accuracies
- `tools_lib.py`: Imports the required libraries to run this project


# Model Overview
3 models were created and trained for this project. `trained_MainCNN` is what will be used going forward as the main model of this project. The other two models (variants) serve as comparison models to the main model. 


## References
1. Original data can be found here [Provenance Information](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?rvi=1)