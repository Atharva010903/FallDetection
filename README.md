Confidence-Based Fall Detection Using Multiple Surveillance Cameras:

📌 Overview
This project implements the Confidence-based Fall Detection (CbFD) approach proposed by Dara Ros and Rui Dai for multi-camera surveillance systems.
It detects falls in video sequences using frame-level features, windowed temporal analysis, and confidence fusion across cameras.

The pipeline extracts handcrafted features from silhouettes, applies a Random Forest classifier to predict fall probability, quantizes it into confidence levels, and fuses results across cameras to reach a final decision.

🚀 Features
Multi-camera fall detection using confidence fusion.
Window-based temporal feature extraction (15-frame sliding window).

Extracted features:
Silhouette ratio & orientation
Centroid height
Optical projection
Brightness
Quality score (LBP-based BRISQUE approximation)
Silhouette size
Random Forest Classifier for fall prediction.
Confidence quantization (11 levels from 0.001 → 1.0).
Confidence-based Decision Fusion (CbFD) across multiple cameras.

Comparison with baseline methods:
Majority Voting
Average Single-Camera

⚙️ Installation
Clone the repository and install dependencies:
git clone https://github.com/Atharva010903/FallDetection.git
cd FallDetection


📝 Usage
1️⃣ Prepare Dataset
Place surveillance videos under data/videos/.
Ensure event labels are in data_tuple3.csv with columns:
chute, cam, start, end, label

2️⃣ Run Notebook
Open the Jupyter Notebook and execute all cells:
jupyter notebook confidence_fall_detection.ipynb

3️⃣ Output
Extracted features → classified into fall / no-fall windows.
Confidence-based fusion → final event prediction.
Results saved in results/cbfd_results.json and confusion matrix plotted.


📊 Evaluation Metrics
The system evaluates:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Sensitivity (Recall) = TP / (TP + FN)
Specificity = TN / (TN + FP)

Also compares performance with:
Majority Voting (baseline)
Average Single-Camera (baseline)
Confidence-based Fusion (CbFD) (proposed)
