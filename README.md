# 🎨 Colourimetric Qualitative Detection using SVM (RBF Kernel)

This project performs qualitative detection of biochemical samples using colorimetric analysis. It leverages computer vision (OpenCV) to extract HSV values from well-plate images and applies a Support Vector Machine (SVM) with RBF kernel to classify samples as:
- ✅ Positive (`+1`)
- ❌ Negative (`-1`)
- ⚠️ Uninterpretable (`*` - skipped during training)

---

## 📌 Objective

Detect whether unknown samples are positive or negative based on color (yellow/pink/orange) from well images, using machine learning.

---

## 🧪 Sample Images

| Well Plate | Labeled Grid |
|------------|--------------|
| ![sample](./data/sample_annotated.png) | + = positive<br>- = negative<br>* = uninterpretable |

---

## 🔍 Workflow Overview

```mermaid
graph TD
    A[Input Image] --> B[Detect Wells & Crop]
    B --> C[Extract HSV Values]
    C --> D[Preprocess Data]
    D --> E[Train SVM Classifier (RBF)]
    E --> F[Predict on Unknown Samples]
    F --> G[Heatmap Visualization]
````

---

## 🛠️ Tech Stack

| Layer              | Tech                                |
| ------------------ | ----------------------------------- |
| 💡 ML Model        | SVM with RBF (scikit-learn)         |
| 🧠 Logic           | HSV Feature Extraction (OpenCV)     |
| 📊 Visualization   | Seaborn, Matplotlib                 |
| 📁 Data Handling   | Pandas, NumPy                       |
| 🖼️ GUI (Optional) | Tkinter (Live Image Classification) |

---

## 📂 Directory Structure

```
Colourimetric_detection_project/
│
├── main.py                 # Main training + prediction script
├── svm_model.ipynb         # Interactive notebook
├── gui_app.py              # GUI to classify unknown samples live (optional)
├── data/
│   ├── sample_1.png        # Image of well plate
│   ├── unknown_sample.png  # Image to test
│   └── well_data_clean.csv # Preprocessed training dataset
├── heatmap.png             # Visualization output
├── requirements.txt
└── README.md
```

---

## 💾 Data Processing

1. **Crop Wells** into a 5×4 or 6×5 grid.
2. **Compute Mean HSV** for each well using OpenCV.
3. Label wells manually using:

   * `+1`: Yellow (positive)
   * `-1`: Pink (negative)
   * `*`: Orange (excluded)
4. Store as CSV (`Row,Col,H,S,V,Label`).

---

## 🔍 ML Logic

* **Feature Vector**: `[H, S, V]` per well
* **Classifier**: `SVC(kernel='rbf', gamma='scale')`
* **Train/Test Split**: 80-20
* **Evaluation**: Accuracy, Confusion Matrix, Heatmaps

---

## 🔥 Heatmap Example

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(predicted_grid, cmap="coolwarm", annot=True)
plt.title("Predicted Sample Class")
```

---

## 🚀 How to Run

### 1. Clone & Install

```bash
git clone https://github.com/your-repo/colourimetric-svm.git
cd colourimetric-svm
pip install -r requirements.txt
```

### 2. Prepare Data

Place all labeled images in `data/` and run:

```bash
python main.py
```

### 3. Optional: Launch GUI

```bash
python gui_app.py
```

---

## 📦 requirements.txt

```
opencv-python
pandas
numpy
scikit-learn
matplotlib
seaborn
tk
```

---

## 📃 One Pager (Project Summary)

* **Title**: *Colourimetric Qualitative Detection using SVM*
* **Objective**: Classify test wells as positive or negative based on colorimetric image detection
* **Input**: Images of micro-well plates
* **Processing**:

  * Grid detection & cropping
  * HSV color extraction
  * Labeling from image map
* **Model**: Support Vector Machine (RBF Kernel)
* **Output**: CSV results, classification grid, heatmap
* **Tech Used**: OpenCV, Scikit-Learn, Pandas, Matplotlib, Seaborn
* **Future Work**: Add semi-supervised learning, deploy via Flask or Streamlit, auto-threshold HSV ranges

---

## 🙌 Contributions

Made with ❤️ by [Subham Bhattacharya](https://github.com/subhambhattacharya)

