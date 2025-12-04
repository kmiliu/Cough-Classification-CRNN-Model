# Corona Hack Respiratory Sound Dataset

## Introduction
A machine learning project for COVID-19 detection using respiratory sounds (breathing, coughing, and counting) collected from volunteers worldwide. This project uses deep learning models (CRNN and baseline ML classifiers) to classify audio samples as healthy or COVID-positive.

> 🚀 **Goal:** Detect COVID-19 infection from respiratory sounds using deep learning  
> 🎧 **Data:** 1,398 samples of breathing, coughing, and counting sounds from global volunteers  
> 🧠 **Model:** CRNN (CNN + BiGRU) trained with Focal Loss to handle class imbalance

### Original / External Datasets
This project uses the following public datasets (raw audio is not included in this repository; please download from the dataset pages):

- [Corona Hack Respiratory Sound Dataset — Kaggle (praveengovi)](https://www.kaggle.com/datasets/praveengovi/coronahack-respiratory-sound-dataset)
- [COVID19 Cough Audio Classification — Kaggle (andrewmvd)](https://www.kaggle.com/datasets/andrewmvd/covid19-cough-audio-classification/data)
- [COVID-19 Cough Sounds — Kaggle (pranaynandan63)](https://www.kaggle.com/datasets/pranaynandan63/covid-19-cough-sounds)

Note: Large media files and dataset directories (audio/video) are excluded from this repository via `.gitignore` (for example: `data/**`, `output/**`, `*.wav`, `*.webm`). If you need a local copy of any dataset, download it from the dataset pages above and place the files under the `data/` directory locally (they are intentionally not tracked here).

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Original Dataset](#original-dataset)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Model Training](#model-training)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)

---

## Project Overview

This project aims to develop an automated COVID-19 detection system using respiratory sound analysis. The approach combines:

- **Audio data collection**: Diverse respiratory sounds from healthy and COVID-positive individuals globally
- **Feature engineering**: Extraction of audio features from raw WAV files
- **Data augmentation**: Synthetic augmentation to increase training data diversity
- **Model training**: CRNN (Convolutional Recurrent Neural Network) and baseline ML models
- **Evaluation**: Comprehensive performance metrics including ROC curves and precision-recall curves

### Key Statistics

- **Metadata file**: `Corona-Hack-Respiratory-Sound-Metadata.csv` (1,398 rows)
- **Audio types**: 9 categories per participant
  - breathing-deep
  - breathing-shallow
  - cough-heavy
  - cough-shallow
  - counting-fast
  - counting-normal
  - vowel-a
  - vowel-e
  - vowel-o
- **Geographic coverage**: Multi-country data (India, Canada, United States, Argentina, etc.)
- **Health status categories**: healthy, covid_positive, respiratory_illness_not_identified, no_respiratory_illness_exposed

---

## Dataset

### Data Structure

```
data/
├── train/          # Training audio samples (organized by collection date)
│   ├── 20200413/
│   ├── 20200415/
│   └── ...
└── test/           # Testing audio samples (organized by collection date)
    ├── 20200803/
    ├── 20200814/
    └── ...
```

### Metadata Fields

The `Corona-Hack-Respiratory-Sound-Metadata.csv` contains:

- **User Information**: USER_ID, COUNTRY, AGE, GENDER, ENGLISH_PROFICIENCY
- **Health Status**: COVID_STATUS, COVID_test_status
- **Medical Conditions**: Diabetes, Asthma, Smoker, Hypertension, Chronic_Lung_Disease, etc.
- **Symptoms**: Fever, Cough, Sore_Throat, Fatigue, Breathing_Difficulties, etc.
- **Audio File Paths**: breathing-deep, breathing-shallow, cough-heavy, cough-shallow, counting-fast, counting-normal, vowel-a, vowel-e, vowel-o

---

## Project Structure

```
CoronaHack-Respiratory-Sound-Dataset/
├── README.md                                    # This file
├── Corona-Hack-Respiratory-Sound-Metadata.csv  # Main metadata file
├── junk_archive.tar.gz                         # Archived experimental code and outputs
│
├── coding/                                      # Source code
│   ├── clean.py                                # Data cleaning utilities
│   ├── filter_audio_quality.py                 # Audio quality filtering
│   ├── build_clean_features.py                 # Feature extraction pipeline
│   ├── generate_augmented_train.py             # Data augmentation
│   ├── generate_model_input_clean.py           # Model input preparation
│   ├── train_crnn_clean7.py                     # CRNN model (final/main)
│   └── train_ml_baseline_clean.py              # Baseline ML models
│
├── data/                                       # Audio dataset
│   ├── train/                                  # Training data (organized by date)
│   └── test/                                   # Testing data (organized by date)
│
└── output/                                     # Generated outputs
    ├── features_train.csv                      # Extracted features (train)
    ├── features_test.csv                       # Extracted features (test)
    ├── model_input_clean_train.csv             # Cleaned model input (train)
    ├── model_input_clean_test.csv              # Cleaned model input (test)
    │
    ├── filtered_audio/                         # Quality-filtered audio files
    │   ├── train/
    │   └── test/
    │
    ├── processed_audio/                        # Preprocessed audio
    │   ├── train/
    │   └── test/
    │
    ├── processed_audio_clean/                  # Augmented + merged training data
    │   ├── train/
    │   └── test/
    │
    ├── processed_audio_augmented/              # Augmented audio samples
    │   └── train/
    │
  ├── crnn_model_results_7/                   # CRNN Model Results (V7 - Final)
  │   ├── best_crnn_focal_biGRU_v7.keras      # Best model weights (V7)
  │   ├── training_curves_crnn_focal_biGRU_v7.png
  │   ├── roc_curve_crnn_focal_biGRU_v7.png
  │   ├── pr_curve_crnn_focal_biGRU_v7.png
  │   ├── test_predictions_f1opt_v7.csv       # Model predictions on test set
  │   └── report.ipynb
  │
    └── filter_log.csv                          # Audio quality filtering log
```

---

## Key Components

### 1. **Data Preprocessing** (`coding/clean.py`)
- Validates and cleans metadata
- Handles missing values
- Ensures data consistency

### 2. **Audio Quality Filtering** (`coding/filter_audio_quality.py`)
- Filters audio files based on quality metrics
- Removes corrupted or low-quality samples
- Outputs quality filtering log

### 3. **Feature Extraction** (`coding/build_clean_features.py`)
- Extracts audio features using librosa
- Creates feature CSVs for model input
- Combines raw and augmented audio data

### 4. **Data Augmentation** (`coding/generate_augmented_train.py`)
- Applies augmentation techniques to training data
- Increases dataset diversity
- Logs augmentation operations

### 5. **Model Input Generation** (`coding/generate_model_input_clean.py`)
- Merges features with metadata
- Prepares input for ML/DL models
- Handles class balancing

-### 6. **Model Training**

#### CRNN Model (`coding/train_crnn_clean7.py`)
- Final Convolutional Recurrent Neural Network used for reported results
- **Model (V7)**: results for the training run are in `output/crnn_model_results_7/`
- Features:
  - Bidirectional GRU layers for temporal modeling
  - Focal loss for handling class imbalance
  - Early stopping and model checkpointing

#### Baseline ML (`coding/train_ml_baseline_clean.py`)
- Traditional ML classifiers for comparison:
  - Logistic Regression
  - Random Forest
  - SVM
  - Gradient Boosting

### 7. **Results & Evaluation**
- ROC curves showing model discrimination ability
- Precision-Recall curves for different thresholds
- Training curves showing convergence
- Classification reports with F1, precision, recall scores

---

## Data Processing Pipeline

### Step 1: Data Collection
Raw audio files organized by collection date under `data/train/` and `data/test/`

### Step 2: Quality Filtering
- Audio files filtered based on quality metrics
- Output: `output/filtered_audio/`

### Step 3: Audio Preprocessing
- Normalization and standardization
- Output: `output/processed_audio/`

### Step 4: Feature Extraction
- Convert audio signals to numerical features
- Output: `features_train.csv`, `features_test.csv`

### Step 5: Data Augmentation (Training Only)
- Create synthetic variations of training data
- Output: `output/processed_audio_augmented/`

### Step 6: Merge & Clean
- Combine original + augmented training data
- Merge with metadata
- Output: `model_input_clean_train.csv`, `model_input_clean_test.csv`

### Step 7: Model Training & Evaluation
- Train CRNN and baseline models
- Generate predictions and evaluation metrics
- Output: Model files, ROC/PR curves, test predictions

---

## Model Training

### CRNN Model (Final - V7)

**Location**: `output/crnn_model_results_7/`

**Architecture**:
- Convolutional layers for spatial feature extraction
- Bidirectional GRU for temporal sequence modeling
- Dense layers for classification

**Training Parameters**:
- Loss function: Focal loss (handles class imbalance)
- Optimizer: Adam
- Batch size: 32 (typical)
- Early stopping: Yes
- Class weights: Computed automatically

**Key Files** (example contents):
- `best_crnn_focal_biGRU_v7.keras` (model weights)
- `training_curves_crnn_focal_biGRU_v7.png` (loss/accuracy)
- `roc_curve_crnn_focal_biGRU_v7.png` (ROC curve)
- `pr_curve_crnn_focal_biGRU_v7.png` (PR curve)
- `test_predictions_f1opt_v7.csv` (test predictions)
- `report.ipynb` (analysis notebook)

### Baseline Models

**Location**: Output from `train_ml_baseline_clean.py`

**Models Included**:
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting

---

## Results

### CRNN Model V7 Highlights

- **Input**: Audio spectral features + augmented training data
- **Output**: Binary classification (COVID / Not COVID)
- **Evaluation Metrics**: 
  - ROC-AUC score
  - Precision, Recall, F1-score
  - Confusion matrix
  
**Visualization Outputs**:
- Training convergence curves
- ROC curve (model discrimination)
- PR curve (precision vs recall trade-off)
- Test set predictions with optimal F1 threshold

### Model Comparison

Baseline ML models provide performance baseline for comparing CRNN effectiveness.

---

## Dependencies

The project requires the following Python libraries:

```
numpy
pandas
scikit-learn
librosa
tensorflow
keras
tqdm
matplotlib
seaborn
```

### Installation

```bash
# Create (if needed) and activate a venv, then install requirements
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo (Streamlit)

A lightweight Streamlit demo is provided to run inference with the V7 CRNN model locally. The demo auto-detects trained PyTorch model files under `output/` (it looks for `crnn_cough.pt` or any `.pt` file) and will prefer a user-uploaded `.pt` model if provided via the UI. The app uses the same feature-extraction pipeline as `coding/build_clean_features.py` and derives imputer/scaler stats from `output/model_input_clean_train.csv` so preprocessing matches training.

Quick start (recommended, run from project root):

1. Activate the project's venv (if not already active):

```bash
source .venv/bin/activate
```

2. Install demo dependencies (if you didn't already):

```bash
pip install -r requirements.txt
```

3. Run the demo:

```bash
streamlit run app.py
```

Notes:
- Place model files (e.g. `crnn_cough.pt`) under the `output/` folder (the app will search and pick the newest `.pt` automatically). Alternatively upload a `.pt` file in the web UI.
- Model binaries are intentionally not tracked in git. If you want to include a model in the repo (not recommended for large files), add it and commit — otherwise keep them local or host in cloud storage and download at runtime.

Docker (optional)

Build and run the demo in Docker (the image does not include model binaries):

```bash
# build image
docker build -t coronahack-demo:latest .

# run (mount project output so model files are readable inside the container)
docker run --rm -p 8501:8501 -v $(pwd)/output:/app/output coronahack-demo:latest
```

Open http://localhost:8501 after the container starts

## Shareable release (quick way to share the demo)

If you want to share a lightweight copy of this project with other collaborators (without datasets or model binaries), use the bundled release helper script which creates a zip that excludes large files and model weights.

1. Make the zip (from the repository root):

```bash
# create a timestamped zip excluding data/, output/, and model binaries
./scripts/make_shareable_release.sh
# or provide a filename
./scripts/make_shareable_release.sh coronahack-demo-share.zip
```

2. Share the generated zip (`coronahack-demo-share-YYYYMMDD-HHMMSS.zip`) via email, Drive, Dropbox, or upload as a GitHub Release asset.

Notes:
- The zip intentionally excludes `data/`, `output/`, `*.pt`, `*.pth`, and other large artifacts to keep the package small and privacy-preserving.
- Instruct recipients how to obtain a model: either copy their own `crnn_cough.pt` into `output/crnn_model_results_7/` or upload a `.pt` via the Streamlit UI after starting the demo.
- The app will not reveal absolute filesystem paths in the UI; only minimal relative names are displayed.

Docker (recommended for reproducible sharing)

If your collaborators have Docker, it's the easiest way to run the demo without installing Python dependencies.

```bash
# build the image (on a machine with Docker)
docker build -t coronahack-demo:latest .

# run the container, mounting local output so model files are visible
docker run --rm -p 8501:8501 -v $(pwd)/output:/app/output coronahack-demo:latest
```

Open http://localhost:8501 in the browser after the container starts. The container image does not include model binaries, so mount or add them into `output/` as shown above.

Privacy reminder

- Do not add large or private model files to the zip. If you need to share a model with trusted collaborators, use secure file sharing (S3 with signed URL, private Drive link, or GitHub Releases) and instruct recipients to place the model into `output/crnn_model_results_7/` before running the demo.
