import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# ---------------------------
# 1. Path setup
# ---------------------------
base_dir = r"C:\Users\surya\.vscode\all in one\bloodcancer1"  # <-- change this path

# ---------------------------
# 2. Data preprocessing
# ---------------------------
img_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ---------------------------
# 3. CNN Model
# ---------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------
# 4. Model training
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop]
)

# ---------------------------
# 5. Save model
# ---------------------------
# Save as HDF5 (recommended)
model.save("blood_cancer_model.h5")
print("Model saved as bloodcancer_model.h5")

# Save as pickle (for backup, less recommended)
with open("blood_cancer_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as bloodcancer_model.pkl")

# ---------------------------
# 6. ROC Curve
# ---------------------------
y_true = []
y_pred = []

val_data.reset()
for i in range(len(val_data)):
    X, y = val_data[i]
    preds = model.predict(X)
    y_true.extend(y)
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

fpr = {}
tpr = {}
roc_auc = {}
n_classes = y_true.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {list(train_data.class_indices.keys())[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curve - Blood Cell Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# ---------------------------
# 7. Model evaluation
# ---------------------------
loss, acc = model.evaluate(val_data)
print(f"Validation Accuracy: {acc*100:.2f}%")
