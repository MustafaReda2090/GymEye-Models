#!/usr/bin/env python
# coding: utf-8

# In[9]:


import mediapipe as mp
import cv2
import numpy as np
import math
import pandas as pd

import pickle

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# In[10]:


IMPORTANT_LMS = [
"NOSE",
"LEFT_SHOULDER",
"RIGHT_SHOULDER",
"LEFT_HIP",
"RIGHT_HIP",
"LEFT_KNEE",
"RIGHT_KNEE",
"LEFT_ANKLE",
"RIGHT_ANKLE",
]

# Generate all columns of the data frame

HEADERS = ["label"] # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


# In[11]:


def extract_important_keypoints(results) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()


def calculate_angle(point1: list, point2: list, point3: list) -> float:
    '''
    Calculate the angle between 3 points
    Unit of the angle will be in Degree
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg


def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


# In[12]:


VIDEO_PATH1 = "E:/FCI Bio/GP/Dataset/Paper/3/3.1/Squat/Labeled_Dataset/videos/32979_1.mp4"
VIDEO_PATH2 = "E:/FCI Bio/GP/Dataset/Paper/3/3.1/Squat/Labeled_Dataset/videos/33029_1.mp4"
VIDEO_PATH3 = "E:/FCI Bio/GP/Dataset/Paper/3/3.1/Squat/Labeled_Dataset/videos/33454_1.mp4"


# In[13]:


# Dump input scaler
with open("./shallow_squat_model/model/input_scaler.pkl", "rb") as f2:
    sh_input_scaler = pickle.load(f2)
    
with open("./knees_inward_model/model/ki_input_scaler.pkl", "rb") as f2:
    ki_input_scaler = pickle.load(f2)
    
with open("./knees_forward_model/model/kf_input_scaler.pkl", "rb") as f2:
    kf_input_scaler = pickle.load(f2)


# In[14]:


from tensorflow.keras.models import load_model
# Load model
shallow_model = load_model("./shallow_squat_model/model/shallow_squat_dp.h5")

inward_model = load_model("./knees_inward_model/model/ki_dp.h5")

forward_model = load_model("./knees_forward_model/model/kf_dp.h5")


# In[18]:


class SquatPoseAnalysis:
    def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float, visibility_threshold: float):
        # Initialize thresholds
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
    
    def get_joints(self, landmarks) -> bool:
        '''
        Check for joints' visibility then get joints coordinate
        '''
        side = self.side.upper()

        # Check visibility
        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_KNEE"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].visibility
        ]
        
        is_visible = all([ vis > self.visibility_threshold for vis in joints_visibility ])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible
        
        # Get joints' coordinates
        self.hip = [landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_HIP"].value].y]
        self.knee = [landmarks[mp_pose.PoseLandmark[f"{side}_KNEE"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_KNEE"].value].y]
        self.ankle = [landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].y]

        return self.is_visible
    
    def analyze_pose(self, landmarks, frame):
        '''
        - Squat Counter
        '''
        self.get_joints(landmarks)

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None)

        # * Calculate curl angle for counter
        squat_angle = int(calculate_angle(self.hip, self.knee, self.ankle))
        if squat_angle > self.stage_down_threshold:
            self.stage = "down"
        elif squat_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        return (squat_angle)


# In[19]:


cap = cv2.VideoCapture(VIDEO_PATH3)

current_stage_sh = ""
current_stage_ki = ""
current_stage_kf = ""
prediction_probability_threshold = 0.6


VISIBILITY_THRESHOLD = 0.65


# Params for counter
STAGE_UP_THRESHOLD = 90
STAGE_DOWN_THRESHOLD = 120


# Init analysis class
left_knee_analysis =SquatPoseAnalysis(side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

right_knee_analysis =SquatPoseAnalysis(side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)




with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        # Reduce size of a frame
        image = rescale_frame(image, 50)

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        if not results.pose_landmarks:
            print("No human found")
            continue

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

        # Make detection
        try:
            
            landmarks = results.pose_landmarks.landmark
            
            (left_squat_angle) = left_knee_analysis.analyze_pose(landmarks=landmarks, frame=image)
            (right_squat_angle) = right_knee_analysis.analyze_pose(landmarks=landmarks, frame=image)

            
            # Extract keypoints from frame for the input
            row = extract_important_keypoints(results)
            X = pd.DataFrame([row, ], columns=HEADERS[1:])
            X = pd.DataFrame(sh_input_scaler.transform(X))
            Y = pd.DataFrame([row, ], columns=HEADERS[1:])
            Y = pd.DataFrame(ki_input_scaler.transform(Y))
            Z = pd.DataFrame([row, ], columns=HEADERS[1:])
            Z = pd.DataFrame(kf_input_scaler.transform(Y))
            

            # Make prediction and its probability
            prediction_sh = shallow_model.predict(X)
            predicted_class_sh = np.argmax(prediction_sh, axis=1)[0]

            prediction_probability_sh = max(prediction_sh.tolist()[0])
            
            
            prediction_ki = inward_model.predict(Y)
            predicted_class_ki = np.argmax(prediction_ki, axis=1)[0]

            prediction_probability_ki = max(prediction_ki.tolist()[0])
            
            
            prediction_kf = forward_model.predict(Z)
            predicted_class_kf = np.argmax(prediction_kf, axis=1)[0]

            prediction_probability_kf = max(prediction_kf.tolist()[0])
            
            
            # Evaluate model prediction
            if predicted_class_sh == 0 and prediction_probability_sh >= prediction_probability_threshold:
                current_stage_sh = "shallow squat"
            elif predicted_class_sh == 1 and prediction_probability_sh >= prediction_probability_threshold: 
                current_stage_sh = "deep squat"
            else:
                current_stage_sh = "UNK"
                
            if predicted_class_ki == 0 and prediction_probability_ki >= prediction_probability_threshold:
                current_stage_ki = "no inward error"
            elif predicted_class_ki == 1 and prediction_probability_ki >= prediction_probability_threshold: 
                current_stage_ki = "knees inward"
            else:
                current_stage_ki = "UNK"
                
            if predicted_class_kf == 0 and prediction_probability_kf >= prediction_probability_threshold:
                current_stage_kf = "no forward error"
            elif predicted_class_kf == 1 and prediction_probability_kf >= prediction_probability_threshold: 
                current_stage_kf = "knees forward"
            else:
                current_stage_kf = "UNK"

                
                
            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (900, 60), (245, 117, 16), -1)
            
            
            cv2.putText(image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_knee_analysis.counter) if right_knee_analysis.is_visible else "UNK", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print(right_knee_analysis.counter)
            
            # Display Left Counter
            cv2.putText(image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_knee_analysis.counter) if left_knee_analysis.is_visible else "UNK", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print(left_knee_analysis.counter)

            # # Display class
            cv2.putText(image, "SHALLOW", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage_sh, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print("SHALLOW:")
            print(current_stage_sh)

            cv2.putText(image, "INWARD", (80, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage_ki, (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print("INWARD:")
            print(current_stage_ki)
            
            
            cv2.putText(image, "FORWARD", (150, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage_kf, (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print("FORWARD:")
            print(current_stage_kf)
            
            # # Display class
            cv2.putText(image, "SH_CLASS", (210, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted_class_sh), (210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print("SH_CLASS:")
            print(predicted_class_sh)           
                
                
            cv2.putText(image, "I_CLASS", (280, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted_class_ki), (280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print("I_CLASS:")
            print(predicted_class_ki)
            
            
            cv2.putText(image, "F_CLASS", (350, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted_class_kf), (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print("SH_CLASS:")
            print(predicted_class_kf)
            
            
            # # Display probability
            cv2.putText(image, "SH_PROB", (420, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(prediction_probability_sh, 2)), (420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print("SH_PROB:")
            print(prediction_probability_sh)
            
            
            cv2.putText(image, "I_PROB", (490, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(prediction_probability_ki, 2)), (490, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print("I_PROB:")
            print(prediction_probability_ki)
            
            
            cv2.putText(image, "F_PROB", (560, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(prediction_probability_kf, 2)), (560, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print("F_PROB:")
            print(prediction_probability_kf)         
                
                
        except Exception as e:
            print(f"Error: {e}")
        
        cv2.imshow("CV2", image)
        
        # Press Q to close cv2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # (Optional)Fix bugs cannot close windows in MacOS (https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
    for i in range (1, 5):
        cv2.waitKey(1)


# In[ ]:




