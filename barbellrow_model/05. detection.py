#!/usr/bin/env python
# coding: utf-8

# In[22]:


import mediapipe as mp
import cv2
import numpy as np
import pandas as pd

import pickle

import warnings
warnings.filterwarnings('ignore')

# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# In[23]:


# Determine important landmarks for barbell row
IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

# Generate all columns of the data frame
HEADERS = ["label"]  # Label column

for lm in IMPORTANT_LMS:
    HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]


# In[35]:


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


# In[36]:


VIDEO_PATH1 = "C:/Users/Alrowad/Exercise-correction/barbellrow_model/dataset/val/Barbell row.mp4"
VIDEO_PATH2 = "C:/Users/Alrowad/Exercise-correction/barbellrow_model/dataset/val/video_2023-05-07_17-51-28.mp4"


# In[37]:


# Load input scaler
with open("./model/input_scaler_lumbar.pkl", "rb") as f:
    input_scaler_lumbar = pickle.load(f)
    
# Load input scaler
with open("./model/input_scaler_torso.pkl", "rb") as f:
    input_scaler_torso = pickle.load(f)


# In[38]:


from tensorflow.keras.models import load_model
# Load model
lumbar_model = load_model("./model/barbell_lumbar_dp.h5")
torso_model = load_model("./model/barbell_torso_dp.h5")


# In[41]:


class BarbellPoseAnalysis:
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
        joints_visibility = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility ]

        is_visible = all([ vis > self.visibility_threshold for vis in joints_visibility ])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible
        
        # Get joints' coordinates
        self.shoulder = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y ]
        self.elbow = [ landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y ]
        self.wrist = [ landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y ]

        return self.is_visible
    
    def analyze_pose(self, landmarks, frame):
        '''
        - Barbellrow Counter
        '''
        self.get_joints(landmarks)

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None)

        # * Calculate curl angle for counter
        barbell_row_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if barbell_row_angle > self.stage_down_threshold:
            self.stage = "down"
        elif barbell_row_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
        
        return (barbell_row_angle)


# In[43]:


cap = cv2.VideoCapture(VIDEO_PATH2)
current_stage_L = ""
current_stage_T = ""
prediction_probability_threshold = 0.6

VISIBILITY_THRESHOLD = 0.65


# Params for counter
STAGE_UP_THRESHOLD = 90
STAGE_DOWN_THRESHOLD = 120


# Init analysis class
left_arm_analysis = BarbellPoseAnalysis(side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

right_arm_analysis = BarbellPoseAnalysis(side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)


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
            
            (left_barbell_row_angle) = left_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
            (right_barbell_row_angle) = right_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)

            
            
            # Extract keypoints from frame for the input
            row = extract_important_keypoints(results)
            X = pd.DataFrame([row, ], columns=HEADERS[1:])
            Y = pd.DataFrame([row, ], columns=HEADERS[1:])
            X = pd.DataFrame(input_scaler_lumbar.transform(X))
            Y = pd.DataFrame(input_scaler_torso.transform(Y))
            

            # Make prediction and its probability
            prediction_L = lumbar_model.predict(X)
            predicted_class_L = np.argmax(prediction_L, axis=1)[0]

            prediction_probability_L = max(prediction_L.tolist()[0])
            
            
            prediction_T = torso_model.predict(Y)
            predicted_class_T = np.argmax(prediction_T, axis=1)[0]

            prediction_probability_T = max(prediction_T.tolist()[0])
            


            # Evaluate model prediction
            # Evaluate model prediction
            if predicted_class_L == 0 and prediction_probability_L >= prediction_probability_threshold:
                current_stage_L = "LC"
            elif predicted_class_L == 1 and prediction_probability_L >= prediction_probability_threshold: 
                current_stage_L = "LE"
            else:
                current_stage_L = "UNK"
                
            if predicted_class_T == 0 and prediction_probability_T >= prediction_probability_threshold:
                current_stage_T = "TC"
            elif predicted_class_T == 1 and prediction_probability_T >= prediction_probability_threshold: 
                current_stage_T = "TE"
            else:
                current_stage_T = "UNK"
                
                
            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (900, 60), (245, 117, 16), -1)
            
             # Display probability
            cv2.putText(image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_arm_analysis.counter) if right_arm_analysis.is_visible else "UNK", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print(right_arm_analysis.counter)
            
            # Display Left Counter
            cv2.putText(image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_arm_analysis.counter) if left_arm_analysis.is_visible else "UNK", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            print(left_arm_analysis.counter)
            
            # # Display class
            cv2.putText(image, "LUMBAR", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage_L, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, "TORSO", (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, current_stage_T, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # # Display class
            cv2.putText(image, "L_CLASS", (165, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted_class_L), (160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, "T_CLASS", (225, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted_class_T), (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # # Display probability
            cv2.putText(image, "L_PROB", (300, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(prediction_probability_L, 2)), (295, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, "T_PROB", (380, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(prediction_probability_T, 2)), (375, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            
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




