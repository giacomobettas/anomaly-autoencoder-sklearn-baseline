# 🧠 Autoencoder-Based Anomaly Detection (scikit-learn Baseline)

This repository contains an early iteration of a video-based anomaly detection project for elderly home video surveillance, developed as part of my **thesis internship for the Master in Data Analysis for Business Intelligence and Data Science**.

The goal of this iteration is to implement a **lightweight, interpretable baseline** using:

- **scikit-learn** for an autoencoder built on `MLPRegressor`
- **OpenCV** for image loading and resizing
- **NumPy** for array operations
- **Matplotlib** for visualizing reconstruction errors

Later iterations of the project (not in this repo) extended the idea with deep learning, data generators, YOLO-based person detection, and face recognition.  
This repository focuses deliberately on the first version as a didactic baseline.

---

## 🔧 Core Idea

1. Images are organized by **person** in `train/` and `test/` folders.
2. A scikit-learn **autoencoder** learns to reconstruct "normal" frames.
3. The **reconstruction error** (mean squared error per frame) is used as an anomaly score.
4. By analyzing reconstruction error distributions, we can:
   - Detect frames that behave differently from the training data.
   - Compute **per-person anomaly rates** and compare them.

---

## 📁 Repository Structure

```text
anomaly-autoencoder-sklearn-baseline/
├─ src/
│  ├─ __init__.py              # Marks src as a package
│  ├─ data_loader.py           # Load images from per-person folders (train/test)
│  ├─ preprocess.py            # Min-max scaling and flattening of images
│  ├─ model.py                 # SklearnAutoencoder wrapper around MLPRegressor
│  ├─ train.py                 # Train autoencoder on images under data/train
│  ├─ evaluate.py              # Evaluate reconstruction errors on data/test
│  └─ infer.py                 # Infer anomaly score for a single image
│
├─ data/
│  └─ README.md                # Expected train/test per-person structure
│
├─ tests/
│  └─ test_smoke.py            # Smoke test for the autoencoder pipeline
│
├─ notebooks/
│  └─ demo_colab.ipynb         # (Optional) Colab notebook to run the pipeline
│
├─ models/                     # (created at runtime) saved autoencoder and scaler
├─ requirements.txt            # Python dependencies (scikit-learn, OpenCV, etc.)
├─ .gitignore                  # Ignore caches, venvs, models, logs
└─ README.md                   # This file
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/giacomobettas/anomaly-autoencoder-sklearn-baseline.git
cd anomaly-autoencoder-sklearn-baseline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Basic Usage (Local)

### 1. Prepare the dataset

Follow the structure described in `data/README.md`, for example:

```text
data/
├─ train/
│  ├─ person1/
│  ├─ person2/
└─ test/
   ├─ person1/
   ├─ person2/
```

Each folder contains `.jpg` / `.png` images. Resolution doesn’t matter: they will be resized to 64×64 grayscale.

### 2. Train the autoencoder

```bash
python -m src.train --train_dir data/train \
    --model_path models/autoencoder.pkl \
    --scaler_path models/scaler.pkl
```

### 3. Evaluate on test set

```bash
python -m src.evaluate --test_dir data/test \
    --model_path models/autoencoder.pkl \
    --scaler_path models/scaler.pkl \
    --threshold_percentile 99.0
```

This prints:
- Global anomaly rate
- Per-person anomaly rate
- A histogram of reconstruction errors with the chosen threshold

### 4. Single-image inference

```bash
python -m src.infer \
    --image_path path/to/image.jpg \
    --model_path models/autoencoder.pkl \
    --scaler_path models/scaler.pkl \
    --threshold 0.01
```

---

## 💻 Google Colab Usage

A ready-to-use Colab notebook is provided under `notebooks/demo_colab.ipynb`.

Typical workflow inside Colab:

1. Mount Google Drive (optional) if your dataset is in Drive.
2. Clone this repository inside Colab.
3. Install dependencies from `requirements.txt`.
4. Set dataset paths (`data/train`, `data/test`) according to your setup.
5. Run the training and evaluation commands directly from the notebook.

See the notebook cells and comments for a step-by-step explanation.

---

## 🧪 Testing

To run the smoke test:

```bash
pytest tests/
```

This checks that the autoencoder can be built, trained briefly, and used for reconstruction on a tiny random dataset.

---

## 📚 Notes

This repository represents a **baseline iteration** from my thesis project in the **Master in Data Analysis for Business Intelligence and Data Science**.

It is intentionally simple and scikit-learn–based, focusing on clarity and structure.

More advanced deep learning–based versions (with YOLO and face recognition) are part of later iterations and will live in separate repositories.
