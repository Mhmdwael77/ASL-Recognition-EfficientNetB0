# 🤟 ASL Real-Time Recognition — EfficientNetB0

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey?style=for-the-badge)

> Real-time American Sign Language alphabet recognition powered by EfficientNetB0, MediaPipe, OpenCV, and Streamlit.

## 📌 Overview

**ASL Real-Time Recognition — EfficientNetB0** is a computer vision project for **real-time American Sign Language (ASL) alphabet recognition** using webcam input.  
The system combines **EfficientNetB0 transfer learning**, **MediaPipe hand tracking**, and a **Streamlit interface** to recognize and translate hand signs into live text.

The model was trained on **87,000 images** across **29 classes**:

- `A-Z`
- `space`
- `del`
- `nothing`

Designed for strong academic and professional presentation, this project demonstrates the full pipeline from **deep learning training and evaluation** to **real-time deployment and lightweight TFLite inference**.

## ✨ Features

- 🎯 Real-time ASL alphabet recognition from webcam feed
- 🧠 EfficientNetB0-based transfer learning with **two-phase training**
- ✋ MediaPipe hand landmark detection and hand ROI extraction
- 🔤 Recognition of **29 classes**: `A-Z + space + del + nothing`
- 🖥️ Streamlit UI with a live **sentence builder**
- ⌨️ Manual controls for **space**, **delete**, and **clear**
- ⚡ TFLite support for faster CPU-friendly inference
- 📈 Strong evaluation results with **96.29% validation accuracy** and **99.97% ROC-AUC**
- 🧪 Modular codebase for inference, conversion, and application runtime

## 📊 Model Performance

| Stage | Metric | Result | Notes |
|---|---|---:|---|
| Phase 1 | Validation Accuracy | **93.10%** | Initial transfer learning stage |
| Phase 2 | Validation Accuracy | **96.76%** | Fine-tuning stage |
| Final Model | Validation Accuracy | **96.29%** | Final validation result |
| Final Model | ROC-AUC | **99.97%** | Excellent class separability |

### 🏆 Class Highlights

| Category | Class | Result |
|---|---|---:|
| Best Performing Class | `F` | **100% F1-score** |
| Most Challenging Class | `X` | **81% accuracy** |

## 🗂️ Project Structure

```text
ASL-Recognition-EfficientNetB0/
├── model/
│   ├── best_model.h5
│   └── best_model.tflite
├── src/
│   ├── app.py
│   ├── inference.py
│   ├── hand_detector.py
│   └── convert_to_tflite.py
├── phase-2.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mhmdwael77/ASL-Recognition-EfficientNetB0.git
cd ASL-Recognition-EfficientNetB0
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

### Run the Streamlit application

```bash
streamlit run src/app.py
```

### Optional: Convert the trained model to TFLite

```bash
python src/convert_to_tflite.py
```

## 📥 Model Download

> **Note:** If the trained model files are not available locally, download them from Kaggle and place them inside the `model/` directory.

**Kaggle model link:**  
`https://www.kaggle.com/<your-model-download-link>`

Expected files:

```text
model/
├── best_model.h5
└── best_model.tflite
```

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Backbone Model | EfficientNetB0 |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Interface | Streamlit |
| Deployment Format | TensorFlow Lite (TFLite) |
| Numerical Computing | NumPy |

## 👨‍💻 Author

**Mohamed Wael**

- LinkedIn: [linkedin.com/in/mohamed-elsiley](https://linkedin.com/in/mohamed-elsiley)
- GitHub: [github.com/Mhmdwael77](https://github.com/Mhmdwael77)
