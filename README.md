# ♻️ Garbage Classification System

An AI-powered waste classification system using Deep Learning and Transfer Learning that classifies garbage images into **7 categories**: Cardboard, Glass, Metal, Organic, Paper, Plastic, and Trash.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌟 Features

- 📤 **Image Upload** — Drag-and-drop waste image classification
- 🎯 **High Accuracy** — 85%+ accuracy using MobileNetV2 transfer learning
- 📊 **Confidence Scores** — Visual confidence breakdown for all categories
- 📋 **Disposal Guidance** — Proper disposal instructions for each waste type
- 📈 **Analytics Dashboard** — Track prediction history with interactive charts
- 🧠 **Multiple Models** — Custom CNN, MobileNetV2, ResNet50, EfficientNetB0
- ⚡ **Fast Inference** — Optimized model for real-time classification (~25ms)
- 🔒 **Input Validation** — File type, size, and integrity checks

---

## 📁 Project Structure

```
garbage_classification/
├── config/
│   └── config.yaml              # All hyperparameters & settings
├── src/
│   ├── data/
│   │   ├── dataset.py           # Data loading, splitting, augmentation
│   │   └── download.py          # Dataset download helper
│   ├── models/
│   │   ├── cnn_model.py         # Custom CNN baseline
│   │   ├── transfer_model.py    # MobileNetV2, ResNet50, EfficientNet
│   │   └── model_factory.py     # Factory pattern for model creation
│   ├── training/
│   │   ├── trainer.py           # Training with callbacks & 2-phase TL
│   │   └── evaluator.py         # Metrics, confusion matrix, reports
│   ├── inference/
│   │   └── predictor.py         # Single image prediction pipeline
│   └── utils/
│       ├── logger.py            # Logging configuration
│       ├── validators.py        # Image validation & security
│       └── helpers.py           # Config loading, seed, utilities
├── app/
│   ├── streamlit_app.py         # Main Streamlit dashboard
│   ├── pages/
│   │   ├── 1_📊_Analytics.py    # Analytics & history page
│   │   └── 2_📖_About.py        # Project info page
│   └── components/
│       ├── sidebar.py           # Sidebar component
│       └── charts.py            # Plotly chart components
├── scripts/
│   ├── train.py                 # CLI training entry point
│   ├── evaluate.py              # CLI evaluation entry point
│   └── predict.py               # CLI prediction tool
├── data/                        # Dataset directories
├── models/saved/                # Trained model files
├── outputs/                     # Plots, reports, logs
├── tests/                       # Test suite
├── docs/                        # Documentation
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Clone the repository
git clone <repository-url>
cd garbage_classification

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: Automatic (Kaggle API)**

```bash
# Set up Kaggle API credentials first
# Place kaggle.json in ~/.kaggle/
python -c "from src.data.download import setup_dataset; setup_dataset()"
```

**Option B: Manual Download**

1. Download from [Kaggle: Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
2. Extract to `data/raw/`
3. Run organization: `python -c "from src.data.download import setup_dataset; setup_dataset()"`

### 3. Train a Model

```bash
# Train MobileNetV2 (recommended)
python scripts/train.py --model mobilenetv2

# Quick test (2 epochs, for verifying setup)
python scripts/train.py --model mobilenetv2 --quick-test

# Train all models for comparison
python scripts/train.py --model all

# Custom training
python scripts/train.py --model resnet50 --epochs 30 --batch-size 16
```

### 4. Evaluate

```bash
python scripts/evaluate.py --model-path models/saved/mobilenetv2_final.h5
```

### 5. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

### 6. CLI Prediction

```bash
python scripts/predict.py --image path/to/waste_image.jpg
```

---

## 🧠 Models

| Model | Parameters | Expected Accuracy | Inference Time |
|-------|-----------|-------------------|---------------|
| Custom CNN | ~2M | 75-80% | ~15ms |
| **MobileNetV2** | ~3.5M | **88-92%** | ~25ms |
| ResNet50 | ~25M | 85-90% | ~50ms |
| EfficientNetB0 | ~5.3M | 87-91% | ~35ms |

### Training Strategy

**Two-Phase Transfer Learning:**
1. **Phase 1 (Feature Extraction):** Freeze base model, train classification head (LR=0.001, 10 epochs)
2. **Phase 2 (Fine-Tuning):** Unfreeze top 30% of base, low LR training (LR=0.0001, 40 epochs)

**Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger

---

## 📊 Evaluation Metrics

The system evaluates models using:
- **Accuracy** — Overall classification accuracy
- **Precision** — Per-class and macro precision
- **Recall** — Per-class and macro recall
- **F1-Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Detailed error analysis

All metrics are generated automatically and saved to `outputs/reports/` and `outputs/plots/`.

---

## 🗂️ Waste Categories

| Category | Icon | Bin | Examples |
|----------|------|-----|----------|
| Cardboard | 📦 | ♻️ Blue | Boxes, packaging |
| Glass | 🫙 | ♻️ Green | Bottles, jars |
| Metal | 🥫 | ♻️ Blue | Cans, aluminum |
| Organic | 🍂 | 🟤 Brown | Food scraps, yard waste |
| Paper | 📄 | ♻️ Blue | Newspapers, mail |
| Plastic | 🧴 | ♻️ Blue/Yellow | Bottles, containers |
| Trash | 🗑️ | ⬛ Black | Non-recyclable waste |

---

## ⚙️ Configuration

All settings are centralized in `config/config.yaml`:

```yaml
# Key settings you might want to modify:
training:
  phase1:
    epochs: 10
    learning_rate: 0.001
  phase2:
    epochs: 40
    learning_rate: 0.0001
  early_stopping:
    patience: 10

image:
  width: 224
  height: 224

model:
  default: "mobilenetv2"
```

---

## 🔒 Security

- File type validation (JPG, PNG, BMP, WEBP only)
- File size limit (10 MB max)
- Image integrity verification
- Dimension validation (32px–10,000px)
- EXIF data stripping

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Dataset:** [Kaggle Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- **Pre-trained Models:** TensorFlow/Keras Applications (ImageNet)
- **UI Framework:** Streamlit
- **Visualization:** Plotly, Matplotlib, Seaborn

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
