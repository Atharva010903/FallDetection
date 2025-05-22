#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


get_ipython().system('pip install scikit-image')


# In[4]:


import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.feature import local_binary_pattern
from skimage.measure import moments, moments_central
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[5]:


WINDOW_SIZE = 15
DATA_PATH = "/kaggle/input/multiple-cameras-fall-dataset/"  # Main dataset folder
DATASET_FOLDER = os.path.join(DATA_PATH, "dataset", "dataset")  # Path to actual dataset videos with chute folders
CSV_PATH = os.path.join(DATA_PATH, "data_tuple3.csv")  # Path to the CSV file
MODEL_PATH = "./models/"  # Path to save models (in the Kaggle working directory)
RESULTS_PATH = "./results/"  # Path to save results (in the Kaggle working directory)


# In[6]:


os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


# In[7]:


def read_dataset_info(csv_path):
    """Read the dataset information from the CSV file"""
    df = pd.read_csv(csv_path)
    # Add debug statements right after reading
    print("CSV data sample:")
    print(df.head())
    print(f"Fall events in CSV: {len(df[df['label'] == 1])}")
    print(f"Unique chute values in CSV: {df['chute'].unique()}")
    return df


# In[8]:


def extract_bounding_box(frame, background_subtractor):
    """Extract bounding box of the human silhouette"""
    fgmask = background_subtractor.apply(frame)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, fgmask
    
    # Find the largest contour (assumed to be the person)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return (x, y, w, h), fgmask

def calculate_silhouette_ratio(bbox):
    """Calculate the ratio of the silhouette's bounding box (width/height)"""
    if bbox is None:
        return None
    x, y, w, h = bbox
    return w / h if h > 0 else 0

def calculate_orientation(fgmask, bbox):
    """Calculate the orientation of the silhouette"""
    if bbox is None or fgmask is None:
        return None
    
    x, y, w, h = bbox
    # Extract the silhouette region
    silhouette = fgmask[y:y+h, x:x+w]
    
    # Calculate moments
    m = moments_central(silhouette)
    
    # Calculate orientation
    # Check if denominator would be zero
    if abs(m[2, 0] - m[0, 2]) < 1e-10:
        return 0
    
    # Calculate the orientation using central moments
    theta = 0.5 * np.arctan2(2 * m[1, 1], (m[2, 0] - m[0, 2]))
    return theta

def calculate_centroid_height(bbox, frame_height):
    """Calculate the height of the centroid from the bottom of the frame"""
    if bbox is None:
        return None
    
    x, y, w, h = bbox
    centroid_y = y + h/2
    return frame_height - centroid_y

def calculate_optical_projection(fgmask):
    """Calculate the projection of the silhouette on the optical x-axis"""
    if fgmask is None:
        return None
    
    # Sum pixels along columns to get x-axis projection
    x_projection = np.sum(fgmask, axis=0)
    return x_projection

def calculate_brightness(frame):
    """Calculate the average brightness of the frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calculate_blind_quality_score(frame):
    """Calculate blind quality score using BRISQUE-like features"""
    # This is a simplified version as full BRISQUE requires a pre-trained model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate LBP
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    
    # Calculate histogram of LBP
    hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    # Calculate simple statistics
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    skewness = skew(gray.flatten())
    kurt = kurtosis(gray.flatten())
    
    # Simple quality score (lower is better)
    score = std_val * (skewness ** 2) * (kurt ** 2)
    
    return score

def calculate_silhouette_size(bbox):
    """Calculate the size of the silhouette"""
    if bbox is None:
        return 0
    
    x, y, w, h = bbox
    return w * h


# In[9]:


def extract_features_from_window(frames, background_subtractor):
    """Extract features from a window of frames following the paper methodology"""
    if len(frames) < WINDOW_SIZE:
        return None
    
    # Arrays for storing frame-by-frame features
    silhouette_ratios = []
    orientations = []
    centroid_heights = []
    optical_projections = []
    brightness_values = []
    quality_scores = []
    silhouette_sizes = []
    
    # Process each frame
    for frame in frames:
        frame_height = frame.shape[0]
        bbox, fgmask = extract_bounding_box(frame, background_subtractor)
        
        # Calculate features only if silhouette is detected
        if bbox is not None and fgmask is not None:
            silhouette_ratio = calculate_silhouette_ratio(bbox)
            orientation = calculate_orientation(fgmask, bbox)
            centroid_height = calculate_centroid_height(bbox, frame_height)
            optical_projection = calculate_optical_projection(fgmask)
            brightness = calculate_brightness(frame)
            quality_score = calculate_blind_quality_score(frame)
            silhouette_size = calculate_silhouette_size(bbox)
            
            # Add values to arrays
            if silhouette_ratio is not None:
                silhouette_ratios.append(silhouette_ratio)
            if orientation is not None:
                orientations.append(orientation)
            if centroid_height is not None:
                centroid_heights.append(centroid_height)
            if optical_projection is not None:
                # Store mean of projection for simplicity
                optical_projections.append(np.mean(optical_projection))
            
            brightness_values.append(brightness)
            quality_scores.append(quality_score)
            silhouette_sizes.append(silhouette_size)
    
    # Return None if insufficient data
    if len(silhouette_ratios) < 2 or len(orientations) < 2:
        return None
    
    # Calculate change rates exactly as described in the paper
    # Change rate of silhouette ratio (first and last frames)
    delta_r = (silhouette_ratios[-1] - silhouette_ratios[0]) / WINDOW_SIZE
    
    # Change rate of orientation
    delta_ort = (orientations[-1] - orientations[0]) / WINDOW_SIZE
    
    # Calculate remaining features as in the paper
    mean_ch = np.mean(centroid_heights) if centroid_heights else 0
    std_ch = np.std(centroid_heights) if centroid_heights else 0
    
    mean_op = np.mean(optical_projections) if optical_projections else 0
    std_op = np.std(optical_projections) if optical_projections else 0
    
    mean_luma = np.mean(brightness_values)
    mean_qs = np.mean(quality_scores)
    mean_size = np.mean(silhouette_sizes)
    
    # Combine features
    features = {
        'delta_r': delta_r,
        'delta_ort': delta_ort,
        'mean_ch': mean_ch,
        'std_ch': std_ch,
        'mean_op': mean_op,
        'std_op': std_op,
        'mean_luma': mean_luma,
        'mean_qs': mean_qs,
        'mean_size': mean_size
    }
    
    return features


# In[10]:


def process_video(video_path, labels_df, chute_id, cam_id):
    """Process a video file and extract features for each window"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Normalize chute ID for consistent filtering
    chute_num = chute_id.replace('chute', '') if 'chute' in chute_id else chute_id
    
    # Ensure both chute and cam match
    video_labels = labels_df[
        (
            (labels_df['chute'] == chute_id) |
            (labels_df['chute'] == int(chute_num)) |
            (labels_df['chute'].astype(str) == chute_num)
        ) &
        (labels_df['cam'] == cam_id)
    ]

    # Create background subtractor with parameters matching the paper
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    all_features = []
    all_labels = []
    detected_fall_events = set()
    
    window_frames = []
    frame_idx = 0
    
    with tqdm(total=frame_count, desc=f"Processing {os.path.basename(video_path)}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
                
            window_frames.append(frame)
            
            if len(window_frames) >= WINDOW_SIZE:
                features = extract_features_from_window(window_frames, background_subtractor)
                
                # Calculate window boundaries
                window_start_frame = frame_idx - WINDOW_SIZE + 1
                window_end_frame = frame_idx
                
                # Determine if this window contains a fall event
                is_fall = False
                for _, row in video_labels.iterrows():
                    start_frame = row['start']
                    end_frame = row['end']
                    label = row['label']
                    fall_id = f"{chute_id}_{cam_id}_{start_frame}_{end_frame}"
                    
                    # Only consider labeled fall events (label=1)
                    if label == 1:
                        # Check for significant overlap between window and fall event (>50%)
                        overlap_start = max(window_start_frame, start_frame)
                        overlap_end = min(window_end_frame, end_frame)
                        overlap_frames = max(0, overlap_end - overlap_start + 1)
                        overlap_ratio = overlap_frames / WINDOW_SIZE
                        
                        if overlap_ratio > 0.5:  # Require substantial overlap
                            if fall_id not in detected_fall_events:
                                detected_fall_events.add(fall_id)
                                print(f"Fall ID: {fall_id}")
                                print(f"FALL DETECTED in window {window_start_frame}-{window_end_frame}, matching fall at {start_frame}-{end_frame}")
                            is_fall = True
                            break
                
                if features is not None:
                    all_features.append(features)
                    all_labels.append(1 if is_fall else 0)
                
                window_frames.pop(0)
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"Total frames processed: {frame_idx}")
    print(f"Total features extracted: {len(all_features)}")
    print(f"Total unique fall events detected: {len(detected_fall_events)}")
    print(f"Total windows with fall label: {sum(all_labels)}")
    
    return all_features, all_labels, detected_fall_events


# In[11]:


def train_confidence_model(features, labels):
    """Train the confidence prediction model"""
    # Convert feature dictionaries to a DataFrame
    df = pd.DataFrame(features)
    
    # Handle missing values
    df = df.fillna(0)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(df, labels, test_size=0.2, random_state=42)
    
    # Train a random forest classifier (similar to bagged trees mentioned in paper)
    model = RandomForestClassifier(
        n_estimators=30,  # Number of learners as mentioned in paper
        max_depth=None,   # Allow full tree growth
        min_samples_split=2,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Validation accuracy: {accuracy:.4f}")
    
    return model

def predict_confidence(model, features):
    """Predict confidence levels as described in the paper (11 discrete levels)"""
    # Convert features to DataFrame
    df = pd.DataFrame(features)
    df = df.fillna(0)
    
    # Get raw probabilities from the model
    raw_scores = model.predict_proba(df)[:, 1]  # Class 1 (fall) probability
    
    # Define the 11 quantization levels from the paper
    levels = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Quantize each score to the nearest level
    quantized_scores = np.zeros_like(raw_scores)
    for i, score in enumerate(raw_scores):
        # Find the nearest quantization level
        quantized_scores[i] = levels[np.argmin(np.abs(np.array(levels) - score))]
    
    return quantized_scores

def confidence_based_fusion(detections, confidence_scores):
    """
    Fuse detection results based on Equation 5 from the paper:
    CbFD = ∑(CoF * x) where x is 1 for fall, -1 for no fall
    """
    # Convert binary detections to -1/1 as per paper
    x_values = np.where(np.array(detections) == 1, 1, -1)
    
    # Calculate the weighted sum (CbFD) as in Equation 5
    cbfd = np.sum(confidence_scores * x_values)
    
    # Final decision based on sign of CbFD
    return 1 if cbfd > 0 else 0

def evaluate_performance(y_true, y_pred):
    """Evaluate the performance of the fall detection system"""
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate sensitivity (recall)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


# In[12]:


"""Main function to run the fall detection system"""
print(f"Reading dataset from: {DATA_PATH}")

if not os.path.exists(CSV_PATH):
    print(f"Error: CSV file not found at {CSV_PATH}")
    exit()

labels_df = read_dataset_info(CSV_PATH)
print(f"Successfully loaded dataset with {len(labels_df)} entries")

chutes = labels_df['chute'].unique()
print(f"Found {len(chutes)} chutes in the dataset")

training_chutes = ['chute02', 'chute03', 'chute04', 'chute05']  # 4 chutes × 8 cams = 32 videos
testing_chutes = ['chute01', 'chute06']                         # 2 chutes × 8 cams = 16 videos

print(f"Training chutes: {len(training_chutes)}")
print(f"Testing chutes: {len(testing_chutes)}")


# In[ ]:


# # ===== Step 1: Train confidence prediction model =====
# all_training_features = []
# all_training_labels = []

# # Process all training videos and extract features
# for chute in training_chutes:
#     for cam_id in range(1, 9):  # 8 cameras per chute
#         video_path = os.path.join(DATASET_FOLDER, str(chute), f'cam{cam_id}.avi')
#         if os.path.exists(video_path):
#             print(f"Processing {video_path}")
#             features, labels, _ = process_video(video_path, labels_df, chute, cam_id)
#             if features and labels:
#                 all_training_features.extend(features)
#                 all_training_labels.extend(labels)
#                 print(f"Extracted {len(features)} feature windows with {sum(labels)} fall events")
#         else:
#             print(f"Warning: Video not found at {video_path}")

# print(f"Total training features: {len(all_training_features)}")
# print(f"Total fall events in training: {sum(all_training_labels)}")


# In[ ]:


# # Train the confidence prediction model
# if all_training_features and all_training_labels:
#     print("Training confidence model...")
#     confidence_model = train_confidence_model(all_training_features, all_training_labels)

#     # Save the model
#     model_save_path = os.path.join(MODEL_PATH, 'confidence_model.pkl')
#     with open(model_save_path, 'wb') as f:
#         pickle.dump(confidence_model, f)
#     print(f"Model saved to {model_save_path}")
# else:
#     print("Error: No training data available.")
#     exit()


# In[13]:


# Update with your actual notebook name
MODEL_INPUT_PATH = "/kaggle/input/copy-for-separate-cells/models/confidence_model.pkl"

# Load the confidence model
with open(MODEL_INPUT_PATH, 'rb') as f:
    confidence_model = pickle.load(f)

print("Model loaded successfully.")


# In[14]:


print("\nEvaluating fall detection on test chutes...")

# Store results for each fall event
all_event_results = {}
# Process each test chute
for chute in testing_chutes:
    print(f"\nProcessing test chute: {chute}")
    
    # Dictionary to store camera results for this chute
    camera_results = {}

    chute_fall_events = None
    chute_no_fall_events = None
    # Process each camera
    for cam_id in range(1, 9):
        # Ensure both chute and cam match
        chute_num = chute.replace('chute', '')
        # --- FALL EVENTS ---
        chute_fall_events = labels_df[
            (
                (labels_df['chute'] == chute) |
                (labels_df['chute'] == int(chute_num)) |
                (labels_df['chute'].astype(str) == chute_num)
            ) &
            (labels_df['cam'] == cam_id) &
            (labels_df['label'] == 1)
        ].drop_duplicates(subset=['start', 'end'])
        
        print(f"Found {len(chute_fall_events)} fall events in chute {chute}, cam{cam_id}")
        
        # --- NO-FALL EVENTS ---
        chute_no_fall_events = labels_df[
            (
                (labels_df['chute'] == chute) |
                (labels_df['chute'] == int(chute_num)) |
                (labels_df['chute'].astype(str) == chute_num)
            ) &
            (labels_df['cam'] == cam_id) &
            (labels_df['label'] == 0)
        ].drop_duplicates(subset=['start', 'end'])
        
        print(f"Found {len(chute_no_fall_events)} no-fall events in chute {chute}, cam{cam_id}")
        
        video_path = os.path.join(DATASET_FOLDER, chute, f'cam{cam_id}.avi')
        if os.path.exists(video_path):
            features, labels, detected_falls = process_video(video_path, labels_df, chute, cam_id)
            if features and labels:
                # Use our single-camera detection algorithm
                # This could be replaced with any algorithm as mentioned in the paper
                detections = [1 if abs(f['delta_r']) > 0.05 else 0 for f in features]
                confidence_scores = predict_confidence(confidence_model, features)
                
                camera_results[cam_id] = {
                    'detections': detections,
                    'labels': labels,
                    'confidence_scores': confidence_scores,
                    'features': features,
                    'detected_falls': detected_falls
                }

    
    # Process each labeled fall event
    for _, event in chute_fall_events.iterrows():
        fall_start = event['start']
        fall_end = event['end']
        fall_id = f"{chute}_{fall_start}_{fall_end}"
        
        # Collect detection results from each camera
        camera_detections = []
        camera_confidences = []
        
        for cam_id, results in camera_results.items():
            # Check if this fall was detected on this camera
            detected = False
            max_confidence = 0.001  # Minimum confidence level
            
            # Check if this specific fall event was detected
            fall_detector_id = f"{chute}_{cam_id}_{fall_start}_{fall_end}"
            if fall_detector_id in results['detected_falls']:
                detected = True
            
            # Find the highest confidence score for frames in this event
            for i, label in enumerate(results['labels']):
                if label == 1:  # This frame was detected as a fall
                    # Calculate window bounds (assuming labels align with windows)
                    window_idx = i * WINDOW_SIZE
                    window_end = window_idx + WINDOW_SIZE
                    
                    # Check if this window overlaps with the fall event
                    if not (window_end < fall_start or window_idx > fall_end):
                        conf = results['confidence_scores'][i]
                        if conf > max_confidence:
                            max_confidence = conf
            
            camera_detections.append(1 if detected else 0)
            camera_confidences.append(max_confidence)
            # Print the required info
            print(f"Chute: {chute}, Cam: {cam_id} | Max Confidence: {max_confidence:.4f} | "
                  f"Detections: {len(camera_detections)} | Confidences: {len(camera_confidences)}")
        
        # Apply confidence-based fusion only if we have camera results
        if camera_detections:
            # Apply CbFD fusion formula from paper (equation 5)
            detection_values = np.where(np.array(camera_detections) == 1, 1, -1)
            weighted_sum = np.sum(np.array(camera_confidences) * detection_values)
            fused_result = 1 if weighted_sum > 0 else 0
            
            # Store the result for this fall event
            all_event_results[fall_id] = {
                'true_label': 1,  # It's a labeled fall event
                'predicted': fused_result,
                'camera_detections': camera_detections,
                'camera_confidences': camera_confidences,
                'weighted_sum': weighted_sum
            }

    # Process each labeled no-fall event to calculate specificity
    for _, event in chute_no_fall_events.iterrows():
        no_fall_start = event['start']
        no_fall_end = event['end']
        no_fall_id = f"{chute}_no_fall_{no_fall_start}_{no_fall_end}"
        
        # Similar process for no-fall events
        camera_detections = []
        camera_confidences = []
        
        for cam_id, results in camera_results.items():
            detected = False
            max_confidence = 0.001
            
            # Find relevant frames
            for i, label in enumerate(results['labels']):
                window_idx = i * WINDOW_SIZE
                window_end = window_idx + WINDOW_SIZE
                
                if not (window_end < no_fall_start or window_idx > no_fall_end):
                    # This window overlaps with the no-fall event
                    conf = results['confidence_scores'][i]
                    if label == 1:  # Incorrectly detected as fall
                        detected = True
                    if conf > max_confidence:
                        max_confidence = conf
            
            camera_detections.append(1 if detected else 0)
            camera_confidences.append(max_confidence)
        
        if camera_detections:
            # Apply fusion
            detection_values = np.where(np.array(camera_detections) == 1, 1, -1)
            weighted_sum = np.sum(np.array(camera_confidences) * detection_values)
            fused_result = 1 if weighted_sum > 0 else 0
            
            all_event_results[no_fall_id] = {
                'true_label': 0,  # It's a no-fall event
                'predicted': fused_result,
                'camera_detections': camera_detections,
                'camera_confidences': camera_confidences,
                'weighted_sum': weighted_sum
            }


# In[15]:


# Calculate overall performance metrics
true_labels = [result['true_label'] for result in all_event_results.values()]
predicted_labels = [result['predicted'] for result in all_event_results.values()]

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

# Calculate metrics as defined in the paper
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\nFall Detection Performance:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Sensitivity: {sensitivity:.2%}")
print(f"Specificity: {specificity:.2%}")

print("\nConfusion Matrix:")
print(f"True Positives: {tp}, False Negatives: {fn}")
print(f"False Positives: {fp}, True Negatives: {tn}")

print("\nComparison with Baseline Methods:")
print("Algorithm | Accuracy | Sensitivity | Specificity")
print(f"CbFD (Proposed) | {accuracy:.2%} | {sensitivity:.2%} | {specificity:.2%}")
print("Majority Vote [3] | 66.67% | 0.00% | 66.67%")
print("Average Single-Camera [9] | 44.00% | 16.00% | 58.00%")

# Save the results
results = {
    'accuracy': float(accuracy),
    'sensitivity': float(sensitivity),
    'specificity': float(specificity),
    'true_positives': int(tp),
    'false_negatives': int(fn),
    'false_positives': int(fp),
    'true_negatives': int(tn),
    'event_results': {k: {
        'true_label': int(v['true_label']),
        'predicted': int(v['predicted']),
        'weighted_sum': float(v['weighted_sum'])
    } for k, v in all_event_results.items()}
}
import json
results_path = os.path.join(RESULTS_PATH, 'results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {results_path}")


# In[16]:


# Create and save confusion matrix visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(true_labels, predicted_labels)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confidence-based Fall Detection Results')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Fall', 'Fall'])
plt.yticks(tick_marks, ['No Fall', 'Fall'])

# Add text annotations to confusion matrix
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'))
plt.show()


# In[ ]:




