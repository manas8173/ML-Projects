import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
YES_DIR = "yes"   # folder with tumor images
NO_DIR  = "no"    # folder with non-tumor images
IMAGE_SIZE = (160, 160)   # resize all images
EPOCHS = 15
BATCH_SIZE = 32
# ----------------------------------------

# Load images and labels
def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = load_img(path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print("Error loading image:", path, e)
    return images, labels

yes_imgs, yes_labels = load_images_from_folder(YES_DIR, 1)
no_imgs, no_labels   = load_images_from_folder(NO_DIR, 0)

X = np.array(yes_imgs + no_imgs, dtype="float32")
y = np.array(yes_labels + no_labels)

print("Total images:", X.shape[0])
print("Labels distribution:", np.bincount(y))

# Split into train/val/test (60/20/20)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# ---------------- Build CNN ----------------
def build_model(input_shape=(160,160,3)):
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
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_model((*IMAGE_SIZE, 3))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name="auc")])

model.summary()

# ---------------- Train ----------------
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1
)

# ---------------- Evaluate ----------------
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

# ---------------- Save model ----------------
# Save as HDF5 (recommended)
model.save("brain_tumor_model.h5")
print("Model saved as brain_tumor_model.h5")

# Save as pickle (backup)
with open("brain_tumor_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as brain_tumor_model.pkl")

# ---------------- Load model ----------------
loaded_model = load_model("brain_tumor_model.h5")
print("Model loaded from HDF5.")

# ---------------- Predictions & ROC ----------------
y_prob = loaded_model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Brain Tumor Detector")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# ---------------- Classification Report ----------------
y_pred = (y_prob >= 0.5).astype(int)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))
print("Test Accuracy (loaded model):", accuracy_score(y_test, y_pred))

