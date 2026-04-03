# AGENTS.md

## Project Overview

This project aims to build a deep learning system for **speech emotion recognition (SER)**.  
The system will classify human emotions from audio signals using neural networks.

Input: raw audio (microphone or file)  
Output: predicted emotion label (e.g., happy, sad, angry, neutral)

---

## Goals

- Build a reproducible ML pipeline for audio-based emotion classification
- Train and evaluate at least one deep learning model
- Provide an interactive demo (CLI or simple UI)
- Ensure modular and collaborative development

---

## Tech Stack

- Language: Python 3.10+
- ML Framework: PyTorch (preferred) or TensorFlow
- Audio Processing: librosa, numpy, scipy
- Visualization: matplotlib, seaborn
- Optional UI: Streamlit or simple CLI
- Experiment tracking (optional): TensorBoard / Weights & Biases

---

## Dataset

Primary: **[Dusha](https://github.com/salute-developers/golos/tree/master/dusha#dusha-dataset)** (Crowd domain: Russian speech, ~201k wav, 4 emotions `angry` / `sad` / `neutral` / `positive`).  
Unpack `crowd.tar`, set `DUSHA_ROOT`, then run `python -m src.data.build_manifest`.  
Podcast domain has no public raw audio; optional `.npy` features via `features.tar` — see `src/data/podcast_features.py`.  
License: [Dusha license](https://github.com/salute-developers/golos/tree/master/dusha#license).

Data format:
- `.wav` files + `raw_crowd_*.jsonl` markup
- labeled by emotion class

---

## Project Structure
```
project-root/
│
├── data/
│ ├── raw/          # DUSHA_ROOT / crowd.tar contents
│ └── processed/    # manifest.csv, splits/*.csv
│
├── src/
│ ├── config.py
│ ├── data/
│ │ ├── build_manifest.py
│ │ ├── manifest.py
│ │ ├── dataset.py
│ │ ├── preprocessing.py
│ │ └── podcast_features.py
│ ├── features/
│ │ └── audio_features.py
│ ├── models/
│ │ ├── cnn_model.py
│ │ └── mlp_model.py
│ ├── training/
│ │ ├── train.py
│ │ ├── evaluate.py
│ │ └── metrics.py
│ └── inference/
│     └── predict.py
│
├── demo/
│ └── app.py
│
├── notebooks/
├── pyproject.toml
├── requirements.txt
├── README.md
└── AGENTS.md
```

---

## Roles (Agents)

### 1. Data Agent
Responsibilities:
- Download and organize dataset
- Normalize audio (sampling rate, length)
- Split dataset (train/val/test)
- Implement preprocessing pipeline

Outputs:
- Clean dataset
- Data loaders

---

### 2. Feature Engineering Agent
Responsibilities:
- Extract audio features:
  - MFCC
  - Mel-spectrogram
  - Optional: chroma
- Optimize feature representation

Outputs:
- Feature extraction module
- Feature validation scripts

---

### 3. Model Agent
Responsibilities:
- Implement models:
  - MLP (baseline)
  - CNN (main model)
- Define architecture and hyperparameters

Outputs:
- Model classes
- Configurable architecture

---

### 4. Training Agent
Responsibilities:
- Training loop
- Loss function and optimizer
- Logging and checkpoints

Outputs:
- Trained models
- Training logs

---

### 5. Evaluation Agent
Responsibilities:
- Compute metrics:
  - Accuracy
  - Confusion matrix
- Analyze model performance

Outputs:
- Evaluation report
- Plots and metrics

---

### 6. Demo Agent
Responsibilities:
- Build inference pipeline
- Implement real-time or file-based prediction
- Create simple UI (CLI or Streamlit)

Outputs:
- Working demo for presentation

---

## Development Workflow

1. Data preparation
2. Feature extraction
3. Baseline model (MLP)
4. Improved model (CNN)
5. Evaluation and tuning
6. Demo integration

---

## Coding Guidelines

- Follow PEP8
- Use type hints where possible
- Keep modules small and reusable
- Separate data, model, and training logic
- Avoid hardcoding paths

---

## Reproducibility

- Fix random seeds
- Save model checkpoints
- Log hyperparameters

---

## Metrics

Primary:
- Accuracy

Secondary:
- F1-score
- Confusion matrix

---

## Deliverables

- Trained model
- Source code
- Demo application
- Short report

---

## Stretch Goals

- Add LSTM / CRNN model
- Real-time emotion tracking
- Multi-dataset training
- Explainability (e.g., feature importance)
