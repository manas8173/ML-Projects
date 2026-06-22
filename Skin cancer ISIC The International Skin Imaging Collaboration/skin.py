import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- CONFIGURATION SECTION ---
INNER_FOLDER_NAME = 'Skin cancer ISIC The International Skin Imaging Collaboration'
BASE_DATA_PATH = os.path.join(os.getcwd(), INNER_FOLDER_NAME)
TRAIN_DIR = os.path.join(BASE_DATA_PATH, 'Train')
TEST_DIR = os.path.join(BASE_DATA_PATH, 'Test')

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 0

# ----------------------------------------------------------------------
# CLASS WEIGHT CALCULATION
# ----------------------------------------------------------------------
def calculate_class_weights(train_generator):
    """Calculates class weights to handle imbalanced data."""
    print("\n--- Calculating Class Weights for Imbalanced Data ---")
    total_samples = train_generator.samples
    class_indices = train_generator.class_indices
    class_totals = np.bincount(train_generator.classes)
    initial_weights = total_samples / (len(class_totals) * class_totals)
    class_weight_dict = {i: weight for i, weight in enumerate(initial_weights)}
    print("Class Name : Total Samples -> Weight")
    for name, index in class_indices.items():
        print(f"{name} : {class_totals[index]} -> {class_weight_dict[index]:.2f}")
    return class_weight_dict

# ----------------------------------------------------------------------
# DIRECTORY MAP VISUALIZATION
# ----------------------------------------------------------------------
def print_directory_map(directory, max_depth=1, max_files=2):
    """Prints a tree-like visualization of the directory structure and sets NUM_CLASSES."""
    print(f"\n--- Directory Map: {os.path.basename(directory)} ---")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at {directory}")
        return
    global NUM_CLASSES
    class_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    if os.path.basename(directory) == 'Train':
        NUM_CLASSES = len(class_names)
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        if level > max_depth:
            del dirs[:]
            continue
        indent = '│   ' * (level - 1) + ('├── ' if level > 0 else '')
        print(f"{indent}{os.path.basename(root)}/")
        file_count = len(files)
        for i, file in enumerate(files[:max_files]):
            file_indent = '│   ' * level
            print(f"{file_indent}├── {file}")
        if file_count > max_files:
            file_indent = '│   ' * level
            print(f"{file_indent}└── ... ({file_count - max_files} more images)")
        dirs.sort()

# ----------------------------------------------------------------------
# DATA DISTRIBUTION VISUALIZATION
# ----------------------------------------------------------------------
def visualize_data_distribution(directory, title):
    """Creates a bar chart showing the number of images per class."""
    print(f"\n--- Analyzing Data Distribution in {title} ---")
    class_names = sorted([name for name in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, name))])
    image_counts = {}
    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        image_counts[class_name] = len(os.listdir(class_path))
    counts = [image_counts[c] for c in class_names]
    total_images = sum(counts)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color='teal')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (total_images*0.003),
                 yval, ha='center', va='bottom', fontsize=9)
    plt.xlabel("Skin Cancer Class")
    plt.ylabel("Number of Images")
    plt.title(f"Data Distribution: {title} Set (Total: {total_images} images)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# INITIAL SETUP
# ----------------------------------------------------------------------
def initial_setup():
    """Runs checks and visualizations before model building."""
    print(f"Base Data Path: {BASE_DATA_PATH}")
    if not os.path.isdir(TRAIN_DIR):
        print(f"\nFATAL ERROR: Training directory not found at: {TRAIN_DIR}")
        print("Please ensure the folder name in the script matches the folder on disk.")
        return False
    print_directory_map(TRAIN_DIR)
    visualize_data_distribution(TRAIN_DIR, "Training")
    visualize_data_distribution(TEST_DIR, "Testing")
    print(f"\nFound {NUM_CLASSES} classes. Ready to build model.")
    return True

# ----------------------------------------------------------------------
# DATA GENERATORS (Improved Augmentation)
# ----------------------------------------------------------------------
def create_generators(train_dir, test_dir, num_classes):
    """Creates augmented data generators."""
    rescale = 1./255
    train_datagen = ImageDataGenerator(
        rescale=rescale,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=rescale)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    validation_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )
    return train_generator, validation_generator

# ----------------------------------------------------------------------
# MODEL BUILDING & TRAINING (Improved)
# ----------------------------------------------------------------------
def build_and_train_model(num_classes, train_gen, val_gen, class_weights):
    """Builds, compiles, and trains the ResNet50 model with fine-tuning and callbacks."""
    print("\n--- Building Improved ResNet50 Transfer Learning Model ---")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Freeze some layers for stability
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
        ModelCheckpoint('best_skin_cancer_resnet50.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("\n--- Starting Fine-Tuned Model Training ---")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )
    return model, history

# ----------------------------------------------------------------------
# TRAINING VISUALIZATION
# ----------------------------------------------------------------------
def plot_results(history):
    """Plots training and validation metrics."""
    print("\n--- Plotting Training History ---")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------------------------------------------------
# SAVE MODEL
# ----------------------------------------------------------------------
def save_model(model):
    """Saves the trained model."""
    model_path = os.path.join(os.getcwd(), 'skin_cancer_resnet50_model_weighted.h5')
    model.save(model_path)
    print(f"\nModel successfully saved to: {model_path}")

# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if not initial_setup():
        exit()
    try:
        train_generator, validation_generator = create_generators(TRAIN_DIR, TEST_DIR, NUM_CLASSES)
        if train_generator.samples > 0:
            class_weights = calculate_class_weights(train_generator)
            model, history = build_and_train_model(NUM_CLASSES, train_generator, validation_generator, class_weights)
            plot_results(history)
            save_model(model)
        else:
            print("\nFATAL ERROR: Data generators found zero images. Check your folder structure.")
    except Exception as e:
        print(f"\nAn error occurred during model creation or training: {e}")
    print("\nScript finished.")