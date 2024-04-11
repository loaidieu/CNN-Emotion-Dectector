import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pandas as pd
import json

from PIL import Image
import zipfile

import sklearn
import sklearn.preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint, uniform

# progress tracker
from tqdm import tqdm

# Import Seaborn
import seaborn as sns
sns.set(font_scale=1.5) # Increase font size