"""
skin_cancer_cnn_full.py

Full workflow for Skin Cancer Classification:
- Train a CNN
- Save the model
- Load the model back
- Evaluate using classification report, confusion matrix, and ROC curve
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import pickle

# ---------------- CONFIG ----------------
TRAIN_DIR = r"C:\Users\surya\.vscode\all in one\Train"
TEST_DIR  = r"C:\Users\surya\.vscode\all in one\Test"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15
# ----------------------------------------

# ---------------- Data Generators ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Class indices:", test_gen.class_indices)
n_classes = len(test_gen.class_indices)
class_names = list(test_gen.class_indices.keys())

# ---------------- Build CNN ----------------
def build_model(input_shape=(160,160,3), n_classes=2):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

model = build_model((*IMAGE_SIZE, 3), n_classes)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc", multi_label=True)]
)
model.summary()

# ---------------- Train Model ----------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# ---------------- Save Model ----------------
model.save("skin_cancer_model.h5")
with open("skin_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as skin_cancer_model.h5 and skin_cancer_model.pkl")

# ---------------- Load Model ----------------
loaded_model = load_model("skin_cancer_model.h5")
print("Model loaded from skin_cancer_model.h5")

# ---------------- Evaluate ----------------
y_prob = loaded_model.predict(test_gen)
y_true = test_gen.classes
y_pred = np.argmax(y_prob, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# ---------------- ROC Curve ----------------
if n_classes == 2:
    y_prob_class1 = y_prob[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob_class1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Skin Cancer Classifier")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
else:
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    plt.figure(figsize=(7,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC={roc_auc:.4f})")

    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve — Skin Cancer Classifier")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
