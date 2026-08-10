<div align="center">

# 🐱 Cat vs Dog Classifier 🐶
### Deep Learning based Image Classification using CNN

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p>
  <img src="https://img.shields.io/github/stars/Omrawat11/cat-vs-dog-classification?style=flat-square&color=yellow" />
  <img src="https://img.shields.io/github/forks/Omrawat11/cat-vs-dog-classification?style=flat-square&color=blue" />
  <img src="https://img.shields.io/github/last-commit/Omrawat11/cat-vs-dog-classification?style=flat-square" />
  <img src="https://img.shields.io/github/repo-size/Omrawat11/cat-vs-dog-classification?style=flat-square" />
</p>

**A Convolutional Neural Network that learns to tell cats and dogs apart — built, trained, and evaluated end-to-end in a single notebook.**

[Overview](#-overview) • [Demo](#-sample-predictions) • [Architecture](#-model-architecture) • [Results](#-results) • [Setup](#-getting-started) • [Usage](#-usage)

</div>

---

## 📌 Overview

This repo contains a complete **binary image classification pipeline** that classifies images as either a **Cat 🐱** or a **Dog 🐶**, using a Convolutional Neural Network (CNN) trained from scratch (and optionally fine-tuned with transfer learning).

> 💡 A great beginner-to-intermediate project for understanding CNNs, image preprocessing, and the full deep learning workflow — from raw images to a working classifier.

| | |
|---|---|
| 🧠 **Task** | Binary Image Classification |
| 🗂️ **Dataset** | Cats vs Dogs (Kaggle) |
| 🏗️ **Model** | Custom CNN |
| 📊 **Framework** | TensorFlow / Keras |
| 🎯 **Accuracy** | ~XX% on validation set |

---

## ✨ Features

- 🖼️ **End-to-end pipeline** — from raw images → preprocessing → training → evaluation → prediction
- 🔄 **Data Augmentation** (rotation, flip, zoom) to reduce overfitting
- 📉 **Training visualizations** — accuracy & loss curves
- 🧪 **Confusion matrix & classification report** for model evaluation
- 🖼️ **Single-image prediction** support — test on your own pictures
- ⚙️ Clean, well-commented, notebook-first workflow

---

## 🧠 Model Architecture

```
Input Image (150x150x3)
        │
   Conv2D + ReLU + MaxPooling   ──┐
   Conv2D + ReLU + MaxPooling     │  Feature
   Conv2D + ReLU + MaxPooling     │  Extraction
        │                       ──┘
   Flatten
        │
   Dense + Dropout
        │
   Dense (Sigmoid) → Cat / Dog
```

<details>
<summary>📋 <b>Click to view full layer summary</b></summary>

| Layer | Type | Output Shape | Params |
|---|---|---|---|
| conv2d_1 | Conv2D | (148, 148, 32) | 896 |
| maxpool_1 | MaxPooling2D | (74, 74, 32) | 0 |
| conv2d_2 | Conv2D | (72, 72, 64) | 18,496 |
| maxpool_2 | MaxPooling2D | (36, 36, 64) | 0 |
| conv2d_3 | Conv2D | (34, 34, 128) | 73,856 |
| maxpool_3 | MaxPooling2D | (17, 17, 128) | 0 |
| flatten | Flatten | (36992) | 0 |
| dense_1 | Dense | (512) | ~18M |
| dropout | Dropout | (512) | 0 |
| output | Dense (Sigmoid) | (1) | 513 |

*(Update this table with your notebook's actual `model.summary()` output)*

</details>

---

## 📊 Results

<div align="center">

| Metric | Score |
|---|---|
| Training Accuracy | XX% |
| Validation Accuracy | XX% |
| Training Loss | 0.XX |
| Validation Loss | 0.XX |

</div>

**Training curves:**

```
Accuracy                          Loss
   │  ╭──────────                   │╲
   │ ╱                              │ ╲___
   │╱                               │     ╲___
   └────────────► epochs            └────────────► epochs
```
*(Replace with actual `matplotlib` plots exported from the notebook, e.g. `assets/accuracy_plot.png`)*

---

## 🔍 Sample Predictions

| Image | Predicted | Confidence |
|:---:|:---:|:---:|
| 🐱 cat_01.jpg | Cat | 98.2% |
| 🐶 dog_01.jpg | Dog | 96.5% |
| 🐱 cat_02.jpg | Cat | 91.7% |

*(Add real sample output images in an `assets/` or `samples/` folder and embed them here with `![alt](path)`)*

---

## 🗂️ Project Structure

```
cat-vs-dog-classification/
│
├── 📓 cat_vs_dog_classification.ipynb   # Main notebook (training + evaluation)
├── 📁 dataset/                          # Train/validation images (not tracked in git)
├── 📁 assets/                           # Plots, sample predictions
├── 📄 requirements.txt                  # Dependencies
├── 📄 LICENSE
└── 📄 README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Omrawat11/cat-vs-dog-classification.git
cd cat-vs-dog-classification
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Core dependencies</summary>

```
tensorflow>=2.x
numpy
matplotlib
pandas
scikit-learn
opencv-python
jupyter
```
</details>

### 3️⃣ Get the dataset
Download the **Dogs vs Cats** dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats) and place it inside the `dataset/` folder.

### 4️⃣ Run the notebook
```bash
jupyter notebook cat_vs_dog_classification.ipynb
```

---

## 🧪 Usage — Predict on a custom image

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("your_image.jpg", target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print("Dog 🐶" if prediction[0] > 0.5 else "Cat 🐱")
```

---

## 🛣️ Roadmap

- [ ] Add Transfer Learning (MobileNetV2 / ResNet50) for higher accuracy
- [ ] Deploy as a Streamlit / FastAPI web app
- [ ] Add Grad-CAM visualizations to explain predictions
- [ ] Export model to TFLite for mobile inference

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/Omrawat11/cat-vs-dog-classification/issues).

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 👤 Author

**Om Rawat**
[GitHub](https://github.com/Omrawat11) 

⭐️ If you found this project useful, consider giving it a star!

</div>
