# Sign-Language-Detection-Model

Welcome to the Sign Language Detection Model project! In a world where effective communication knows no boundaries, my project aims to bridge the gap between sign language and spoken language. This model is designed to empower individuals who use sign language as their primary means of communication by providing real-time detection and interpretation of sign language gestures and actions.

In this project, I have harnessed the power of computer vision and machine learning to create a system that captures and analyzes real-time actions and signs performed by a person. By leveraging cutting-edge technology, we can convert these gestures into spoken or written words, making communication more accessible and inclusive for everyone.

What you can find here
- Project code
- Model Trained Data
- Model Simulations
- Model Stats
- Technical Report

## Content Overview
- Install and Import Dependencies
- KeyPoints using MP Holistics
- Extract Keypoint Values
- Setup Folders for Collection
- Collect key points values for Training and Testing
- Preprocess data and create labels and Features
- Build and Train LSTM Neural Network
- Make Predictions
- Save Weights
- Perform Real-Time Sign Language using OpenCV

### Install and Import Dependencies

```python
!pip install tensorflow==2.13.0 tensorflow-gpu==2.12.0 opencv-python mediapipe sklearn matplotlib
!pip install scikit-learn scipy matplotlib numpy

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
```




