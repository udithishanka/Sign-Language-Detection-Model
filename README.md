# Sign-Language-Detection-Model

Welcome to the Sign Language Detection Model project! In a world where effective communication knows no boundaries, this project aims to bridge the gap between sign language and spoken language. This model is designed to empower individuals who use sign language as their primary means of communication by providing real-time detection and interpretation of sign language gestures and actions.

In this project, I have harnessed the power of computer vision and machine learning to create a system that captures and analyzes real-time actions and signs performed by a person. By leveraging cutting-edge technology, we can convert these gestures into spoken or written words, making communication more accessible and inclusive for everyone.

What you can find here
- Project code
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
- Test in Real Time

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

In this project, we leverage a set of essential libraries to accomplish our objectives. TensorFlow, a robust deep learning framework, serves as the backbone for neural network modeling. TensorFlow GPU complements this by offering GPU support, accelerating computations when compatible hardware is available. OpenCV plays a crucial role in enabling webcam access, allowing us to process video input effectively. For extracting keypoints and performing various perception tasks, we turn to Mediapipe. scikit-learn (sklearn) comes into play for data splitting, training, and evaluation in machine learning. Finally, matplotlib is our go-to tool for data visualization, enabling us to create insightful graphs and plots. Together, these libraries form the foundation of this project's capabilities, enabling us to achieve our goals efficiently and effectively.

### KeyPoints using MP Holistics

```python
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```
In this section we import the holistic model from mediapipe to detect the key points and the drawing_utils for drawing the landmarks. 
'mediapipe_detection' function takes two arguments: an input image and a MediaPipe model. The purpose of this function is to process the input image using the specified model and return the processed image along with the model's results. 
The steps involved are: 
- Converting the input image from BGR color format to RGB format (MediaPipe's expected format).
- Temporarily setting the image's writeable flag to False to make it read-only for processing.
- Using the model to process the image and make predictions.
- Resetting the image's writeable flag to True.
- Converting the image back to BGR color format.
- Finally, returning the processed image and the model's results.

draw_landmarks function takes an image and the results obtained from the MediaPipe model as input. Its purpose is to draw landmarks on the input image for the face, pose, left hand, and right hand. It utilizes the mp_drawing library to achieve this. 

draw_styled_landmarks function is responsible for drawing landmarks on the input image, but it allows for custom styling of the landmarks. Specifically, it draws landmarks for the face, pose, left hand, and right hand. Each type of landmark is styled differently, with specified colors and line thicknesses to improve visualization. 

Video Capture and Main Loop In this section, the code sets up video capture from the default camera (usually the webcam) using cv2.VideoCapture(0). 
It then enters a main loop that continuously: 
- Captures video frames from the camera feed using cap.read().
- Calls the mediapipe_detection function to process each frame using the holistic model.
- Prints the detected landmarks to the console.
- Utilizes draw_styled_landmarks to draw the detected landmarks on the frame.
- Displays the processed frame with overlaid landmarks using OpenCV.

The final part of the code handles the exit mechanism. The main loop continues running until the user presses the 'q' key. When 'q' is pressed, the loop breaks, and the script releases the camera (cap.release()) and closes all OpenCV windows (cv2.destroyAllWindows()).

![fig2](https://github.com/udithishanka/Sign-Language-Detection-Model/assets/107479890/ac1134e0-40cd-4310-87e9-0ef30545758d)

### Extract Keypoint Values

```python
pose = []
for res in results.pose_landmarks.landmark:
    test = np.array([res.x, res.y, res.z, res.visibility])
    pose.append(test)
pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landma

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

```

This code segment extracts and organizes keypoint data from MediaPipe's holistic model results. It creates numpy arrays to store the 2D and 3D coordinates, along with visibility information, of keypoints related to pose, face, left hand, and right hand. The code iterates through each type of landmark, constructs arrays for each detected landmark, and appends them to their respective arrays. If certain landmarks are not detected, the code populates the corresponding arrays with zeros to maintain consistent data structures. Additionally, a function called extract_keypoints encapsulates this process, taking the results object as input and returning a concatenated array containing all the keypoints.

### Setup Folders for Collection

```python
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30
for action in actions: 
    dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(dirmax+sequence)))
        except:
            pass

```

This code is responsible for setting up the directory structure to store exported data, specifically numpy arrays containing video sequences of actions to be detected. It begins by defining several parameters, including the path where the data will be exported (DATA_PATH), the list of actions to detect (such as 'hello,' 'thanks,' and 'iloveyou'), the number of video sequences to capture (no_sequences), and the length of each video sequence in terms of frames (sequence_length). 
The code then enters a loop to create folders and subfolders for each action and its associated video sequences. It starts with the index specified by start_folder and iterates through the list of actions. For each action, it calculates the maximum directory index within that action's folder and increments it by the sequence number. This ensures that each new video sequence is stored in its own uniquely numbered subfolder within the corresponding action folder. Exception handling is used to ensure that the code doesn't raise errors if the folders already exist.

### Collect Keypoint Values for Training and Testing

```python
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder+no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()

```
In this block of code, we collect the data for training and testing the model. 


### Preprocess Data and Create Labels and Features

```python
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

```

### Build and Train LSTM Neural Network

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()

```
Now comes the good part. We train the deep learning model using TensorFlow and Keras to perform action recognition on video sequences. The architecture of the model consists of a sequential stack of layers. It starts with three Long Short-Term Memory (LSTM) layers with varying units (64, 128, and 64, respectively), all using the ReLU activation function. These LSTM layers are designed for sequence data, and return_sequences=True is set to ensure that they output sequences rather than a single value. The input shape is specified as (30, 1662), which corresponds to the length of each sequence (30 frames) and the number of features per frame (1662). 
Following the LSTM layers, there are three fully connected (Dense) layers. The first Dense layer contains 64 units with ReLU activation, followed by a layer with 32 units and ReLU activation, and finally an output layer with a number of units equal to the number of distinct actions (as determined by actions.shape[0]) and softmax activation. This output layer produces probability scores for each action class, allowing the model to predict the most likely action for a given input sequence. 
The model is compiled with the Adam optimizer and categorical cross-entropy loss, which is commonly used for multi-class classification problems. Additionally, the categorical accuracy metric is specified for monitoring the model's performance during training. The training process uses the fit method, where X_train and y_train are assumed to be the training data and labels, respectively. It trains for a specified number of epochs (2000 in this case) and utilizes a TensorBoard callback for logging training metrics to a specified directory (log_dir) for later visualization and analysis. After training, the model's architecture and summary are displayed below.

![modelsummary](https://github.com/udithishanka/Sign-Language-Detection-Model/assets/107479890/25b11a88-605b-46b4-b3ef-166bd20c99cb)


### Test in Real Time

```python
from scipy import stats
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, actions, image, colors))

```

![hello](https://github.com/udithishanka/Sign-Language-Detection-Model/assets/107479890/53a477d0-cc22-4ae8-8b15-fbcbd8c67843)

![thanks](https://github.com/udithishanka/Sign-Language-Detection-Model/assets/107479890/213e38da-4caf-418f-9b9e-83893d9bee17)

![iloveyou](https://github.com/udithishanka/Sign-Language-Detection-Model/assets/107479890/e2ce25e7-4c33-408c-b3e0-b2aa14bc73b3)











