# Pneumonia Detection from Chest X-rays using ResNet50

This project uses deep learning and transfer learning to detect **Pneumonia** from chest X-ray images using the **ResNet50** architecture. It involves model training, fine-tuning, evaluation with metrics like sensitivity/specificity, and prediction on custom images.

---

## 📁 Dataset

- **Source**: [Paul Mooney's Chest X-ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: `NORMAL` and `PNEUMONIA`
- **Structure**:
  ```
  chest_xray/
    ├── train/
    ├── val/
    └── test/
  ```

---

## 🧠 Model Architecture

- **Base Model**: `ResNet50` pretrained on ImageNet
- **Custom Layers**:
  - `GlobalAveragePooling2D`
  - `Dense(128, relu)`
  - `Dropout(0.5)`
  - `Dense(1, sigmoid)` for binary classification

---

## 🏋️‍♂️ Training

- Data augmentation using `ImageDataGenerator`
- Class imbalance handled with computed **class weights**
- Optimizer: `Adam`
- Loss: `Binary Crossentropy`
- Metrics: `Accuracy`
- Initial Training: 10 epochs (base model frozen)
- Fine-tuning: 5+ epochs (last 30 layers unfrozen, learning rate lowered)

---

## 🔧 Fine-Tuning

- Enabled training on the last 30 layers of the ResNet50 base
- Used a smaller learning rate (`1e-5`)
- Additional training with **augmented data**

---

## 📊 Evaluation Metrics

- **Confusion Matrix**
- **Classification Report**
- **Accuracy**
- **Sensitivity (Recall)**
- **Specificity**

---

## 🔍 Custom Image Prediction

Test the model with your own chest X-ray image:
```python
img_path = '/content/pneu.jpeg'  # Replace with your image
prediction = model.predict(img_array)
```

Displays the prediction (`NORMAL` or `PNEUMONIA`) with confidence.

---

## 💾 Model Saving

Models are saved at each important step:
- `pneumonia_model.h5`
- `resnet50_pneumonia_finetuned.h5`
- `pneumonia_detection_focal_loss.h5`

These can be saved to and loaded from Google Drive.

---

## 🚀 How to Run

1. Clone this repo.
2. Set up environment (TensorFlow, scikit-learn, matplotlib, seaborn).
3. Download dataset via Kaggle API or manually.
4. Train the model using `model.fit(...)`.
5. Evaluate or predict using test set or custom images.

---

## 📌 Requirements

- TensorFlow >= 2.0
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- kagglehub (for downloading dataset)

---



## 📝 License

This project is under the MIT License.

