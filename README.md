# 🏏 Cricket Shot Classification Using Deep Learning

A web-based cricket batting shot classification system built using **EfficientNet-B0, Bidirectional GRU (BiGRU), and Temporal Attention**.

The application allows users to upload a cricket batting video and receive a predicted shot category, model confidence score, and probability distribution across all supported shot classes.

### 🌐 Live Demo

**[Try the Cricket Shot Classification Website](https://cricket-shot-classification-nj8nq2hmcu3nqfszb2s9t5.streamlit.app/)**

### 🎯 Project Objective

To develop a deep learning system capable of identifying different cricket batting shots from video sequences by learning both spatial and temporal information.

### 🧠 Model Architecture

**Video Input → Frame Extraction → EfficientNet-B0 → BiGRU → Temporal Attention → Classification**

- **Frame Extraction:** Uniformly samples 30 frames from each video.
- **EfficientNet-B0:** Extracts spatial features from individual frames using an ImageNet-pretrained backbone.
- **BiGRU:** Learns temporal relationships across the sequence of frame features.
- **Temporal Attention:** Helps the classifier focus on informative parts of the video sequence.
- **Classification Layer:** Predicts one of 10 cricket batting shot categories.

### 🏏 Supported Cricket Shots

1. Cover Drive
2. Defense
3. Flick
4. Hook
5. Late Cut
6. Lofted
7. Pull
8. Square Cut
9. Straight Drive
10. Sweep

### 📊 Model Performance

| Metric | Result |
|---|---|
| Best validation accuracy | 78.85% |
| Test accuracy | **83.27%** |
| Correct test predictions | 234 / 281 |
| Number of classes | 10 |

These results were measured on the project's held-out test dataset. Performance on videos from different sources may vary.

### 💻 Technologies Used

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- BiGRU and Temporal Attention
- OpenCV
- NumPy
- Streamlit
- Git and GitHub

### 📁 Project Structure

```text
cricket-shot-classification/
├── app.py
├── predict.py
├── cricket_shot_prototype_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

### 🚀 Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/nivrithakunduru-web/cricket-shot-classification.git
cd cricket-shot-classification
```

**2. Create a Python environment**

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate`.

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**4. Launch the application**

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

The application may download pretrained EfficientNet-B0 weights during its first run, so an internet connection may be required.

### 🔍 How to Use

1. Open the application.
2. Upload a cricket batting video in MP4, AVI, MOV, or MKV format.
3. Click **Predict Cricket Shot**.
4. View the predicted shot, confidence score, and class probabilities.

**Note:** Confidence is the model's estimated probability, not a guarantee that the prediction is correct.

### ⚠️ Limitations

- Prediction quality depends on the video and its similarity to training data.
- Unfamiliar camera angles, lighting, backgrounds, or batting styles may affect results.
- The model predicts only the 10 supported cricket shot categories.
- The application is an academic prototype, not a professional cricket analytics system.

### 🔮 Future Improvements

- Improve generalization using larger and more diverse datasets.
- Evaluate the model on independent external datasets.
- Compare alternative video classification architectures.
- Improve inference speed and user experience.

### 🔗 Repository

[GitHub — Cricket Shot Classification](https://github.com/nivrithakunduru-web/cricket-shot-classification)
