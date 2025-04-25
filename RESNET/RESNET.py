import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

import kagglehub

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)
base_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray'

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
def count_images(directory):
    return {
        'NORMAL': len(os.listdir(os.path.join(directory, 'NORMAL'))),
        'PNEUMONIA': len(os.listdir(os.path.join(directory, 'PNEUMONIA')))
    }

train_counts = count_images(train_dir)

# Visualize class distribution
sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()))
plt.title("Training Set Distribution")
plt.show()

# Compute class weights
labels = [0]*train_counts['NORMAL'] + [1]*train_counts['PNEUMONIA']
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)




base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # <-- Add this
output = Dense(1, activation='sigmoid')(x)


model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()



history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weights
)


model.save('pneumonia_model.h5')
model.save('/content/drive/MyDrive/pneumonia_model.h5')


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1,2,2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title("Loss")

    plt.show()

plot_history(history)  


def EvaluateOnTestSet():
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {test_acc:.2f} - Loss: {test_loss:.2f}")

EvaluateOnTestSet()  

y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_labels, target_names=["NORMAL", "PNEUMONIA"]))


cm = confusion_matrix(y_true, y_pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()  


from tensorflow.keras.models import load_model
model = load_model('pneumonia_model.h5')

train_datagen_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator_aug = train_datagen_aug.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Validation data (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)



base_model.trainable = True

# Optional: Fine-tune only the last N layers (say, last 30)
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 2. Recompile the model with a low learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # very small lr for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3. Fine-tune the model
fine_tune_epochs = 5
total_epochs = 10 + fine_tune_epochs  # if 10 were already done

history_finetune = model.fit(
    train_generator_aug,
    epochs=total_epochs,
    initial_epoch=10,  # IMPORTANT to continue from epoch 10
    validation_data=val_generator
)


plot_history(history_finetune)
EvaluateOnTestSet()


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Predict classes
y_pred_probs = model.predict(test_generator)
y_pred = np.round(y_pred_probs).astype(int)  # binary classification

# Get true labels
y_true = test_generator.classes

# Print report
print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))


model.save("resnet50_pneumonia_finetuned.h5")
model.save('/content/drive/MyDrive/resnet50_pneumonia_finetuned.h5')


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(class_weights))
print(class_weights)


history_focal_loss = model.fit(
    train_generator_aug,
    epochs=total_epochs,
    initial_epoch=10,
    validation_data=val_generator,
    class_weight=class_weights
)

plot_history(history_focal_loss)
EvaluateOnTestSet()

from sklearn.metrics import classification_report

y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int)

print(classification_report(y_true, y_pred_classes, target_names=['NORMAL', 'PNEUMONIA']))


model.save("pneumonia_detection_focal_loss.h5")
model.save('/content/drive/MyDrive/pneumonia_detection_focal_loss.h5')



from sklearn.metrics import confusion_matrix
y_true = test_generator.classes  # True labels
y_pred = model.predict(test_generator)  # Predictions
y_pred_labels = (y_pred > 0.5).astype(int).flatten() # Convert predictions to binary labels
cm = confusion_matrix(y_true, y_pred_labels)
TN, FP, FN, TP = cm.ravel() # Extract values from confusion matrix

sensitivity = TP / (TP + FN)  # Also known as Recall
specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")



#in order to mount the google drive so that we can store the model there and load it and use it later we should do this
from google.colab import drive
drive.mount('/content/drive')
model.save('/content/drive/MyDrive/pneumonia_model.h5')
from tensorflow.keras.models import load_model
model = load_model('/content/drive/MyDrive/pneumonia_model.h5')



import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 1. Load the trained model
model = load_model('pneumonia_model.h5')  # Ensure this matches your model filename

# 2. Load and preprocess the image
img_path = '/content/pneu.jpeg'  # Your image path
img = image.load_img(img_path, target_size=(224, 224))  # ✅ Corrected syntax here
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize the pixel values

# 3. Predict
prediction = model.predict(img_array)
predicted_class = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

# 4. Display result
plt.imshow(img)
plt.title(f"Prediction: {predicted_class}")
plt.axis('off')
plt.show()

# Optional: Print confidence
print(f"Confidence: {prediction[0][0]:.4f}")

