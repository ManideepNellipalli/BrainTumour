[README.md](https://github.com/user-attachments/files/28925069/README.md)
# 🧠 Brain Tumor Classification 

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.1-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A deep learning model that classifies brain MRI scans into four categories — **Glioma, Meningioma, Pituitary Tumor, and No Tumor** — using transfer learning with EfficientNetB3.

---

## 📌 Live Demo

> 🔗 **[View Live Demo](https://manideepnellipalli.github.io/BrainTumour/)**

---

## 📖 Project Overview

Brain tumors are among the most critical medical conditions requiring fast and accurate diagnosis. This project leverages **transfer learning** with the **EfficientNetB3** architecture pre-trained on ImageNet to classify brain MRI scans into four classes:

| Class | Description |
|---|---|
| Glioma | Tumor arising from glial cells |
| Meningioma | Tumor in the meninges layer |
| Pituitary | Tumor in the pituitary gland |
| No Tumor | Healthy brain scan |

The model is trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

---

## ✨ Features

- Transfer learning with **EfficientNetB3** (ImageNet weights)
- Custom classification head with **L1/L2 regularization** and **Dropout (0.45)** to reduce overfitting
- **Adamax optimizer** with categorical cross-entropy loss
- Evaluation via **confusion matrix** and **classification report** (precision, recall, F1-score)
- Training and validation **loss/accuracy plots**
- Model saved as `.h5` for reuse and inference

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| TensorFlow 2.9.1 / Keras | Model building & training |
| EfficientNetB3 | Pre-trained base model |
| NumPy / Pandas | Data handling |
| OpenCV / PIL | Image loading & preprocessing |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Evaluation metrics, data splitting |

---

## 📂 Project Structure

```
brain-tumor-classification/
│
├── brain_tumor_classification.ipynb   # Main Jupyter notebook
├── brain_tumor_classification.py      # Python script version
├── Brain Tumors.h5                    # Saved trained model
│
├── assets/                            # Images for README & GitHub Pages
│   ├── sample_mri.png
│   ├── training_curves.png
│   └── confusion_matrix.png
│
├── index.html                         # GitHub Pages demo page
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── LICENSE
```

---

## ⚙️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
tensorflow==2.9.1
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
Pillow
```

### 3. Download the Dataset

Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle and place it as:

```
/kaggle/input/brain-tumor-mri-dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

### 4. Run the Notebook

Open `brain_tumor_classification.ipynb` in Jupyter or Kaggle and run all cells.

### 5. Run Inference on a Single Image

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adamax
import tensorflow as tf
from PIL import Image

model = load_model('Brain Tumors.h5', compile=False)
model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

image = Image.open('your_mri_image.jpg').resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
score = tf.nn.softmax(predictions[0])
print(f"Predicted: {class_labels[tf.argmax(score)]}")
```

---

## 🖼️ Screenshots

> *(Add your actual output images to the `assets/` folder and update paths below)*

### Sample MRI Images (Training Data)
![Sample MRI Grid](assets/sample_mri.png)

### Training & Validation Curves
![Training Curves](assets/training_curves.png)

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

---

## 📊 Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB3 (ImageNet weights, include_top=False, pooling='max')
    ↓
BatchNormalization (momentum=0.99, epsilon=0.001)
    ↓
Dense(256, activation='relu', L2=0.016, L1 activity=0.006, L1 bias=0.006)
    ↓
Dropout(0.45)
    ↓
Dense(4, activation='softmax')
```

**Training config:** Adamax (lr=0.001) | Categorical Cross-Entropy | 10 epochs | Batch size 16

---

## 🚀 Future Improvements

- [ ] Add Grad-CAM visualizations to highlight tumor regions in MRI scans
- [ ] Deploy as a web app using **Streamlit** or **Flask**
- [ ] Experiment with other architectures (EfficientNetB5, ResNet50, VGG16)
- [ ] Add data augmentation (rotation, flipping, zoom) to improve generalization
- [ ] Integrate DICOM format support for real clinical MRI files
- [ ] Hyperparameter tuning using Keras Tuner

---

## 👤 Author

**Manideep**
- 💼 Software Engineer | RPA & Python Developer
- 🌐 [LinkedIn](https://www.linkedin.com/in/manideep-nellipalli-3a33b5270/)
- 💻 [GitHub](https://github.com/ManideepNellipalli)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset by [Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) on Kaggle
- EfficientNet paper: *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks* (Tan & Le, 2019)
