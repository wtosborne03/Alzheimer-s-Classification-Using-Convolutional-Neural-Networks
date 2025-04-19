# %% [markdown]
# %% [markdown]<br>
# # Checkpoint 2: Alzheimer's MRI Classification Modeling<br>
# <br>
# This notebook implements two models to classify MRI scans into four stages:<br>
# 1. A Custom Convolutional Neural Network (CNN) built from scratch.<br>
# 2. Transfer Learning using a pre-trained model (MobileNetV2).<br>
# <br>
# We will load the data, build/compile/train each model, and visualize the training history (accuracy and loss).

# %% [markdown]
# %%

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# Or VGG16, ResNet50, etc.
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory

# %%
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Is Metal enabled?", tf.config.experimental.list_physical_devices('GPU')[0].device_type == 'METAL' if tf.config.experimental.list_physical_devices('GPU') else False)

if tf.config.experimental.list_physical_devices('GPU'):
    print("Using GPU acceleration")
    # Optional: Configure GPU memory growth
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("GPU not available, using CPU")

# %%
import matplotlib.pyplot as plt
import numpy as np
import os

# %% [markdown]
# %% [markdown]<br>
# ## 1. Setup and Configuration

# %% [markdown]
# %%<br>
# --- Configuration ---<br>
# !! IMPORTANT: Replace this with the actual path to your dataset !!

# %%
DATASET_DIR = "/Users/williamosborne/.cache/kagglehub/datasets/abdullahtauseef2003/adni-4c-alzheimers-mri-classification-dataset/versions/1/AugmentedAlzheimerDataset"

# %% [markdown]
# Image parameters (based on your Checkpoint 1 EDA)

# %%
IMG_HEIGHT = 190  # [cite: 14]
IMG_WIDTH = 200  # [cite: 14]
IMG_CHANNELS_GRAY = 1  # Grayscale images [cite: 16]
IMG_CHANNELS_RGB = 3  # Required for many pre-trained models
IMG_SHAPE_GRAY = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_GRAY)
IMG_SHAPE_RGB = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS_RGB)

# %% [markdown]
# Training parameters (adjust as needed)

# %%
BATCH_SIZE = 32
EPOCHS_CUSTOM_CNN = 15  # Start with fewer epochs, increase if needed
EPOCHS_TRANSFER = 10    # Fewer epochs often needed for fine-tuning

# %% [markdown]
# Dataset parameters

# %%
CLASS_NAMES = ["AD", "CN", "EMCI", "LMCI"]  # [cite: 3]
NUM_CLASSES = len(CLASS_NAMES)
VALIDATION_SPLIT = 0.2  # Use 20% of data for validation

# %% [markdown]
# %% [markdown]<br>
# ## 2. Load and Prepare Data<br>
# We use `image_dataset_from_directory` to load the images. It automatically infers class labels from the directory structure and splits data into training and validation sets.

# %% [markdown]
# %%<br>
# --- Load Data for Custom CNN (Grayscale) ---

# %%
train_ds_gray, val_ds_gray = image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='categorical',  # for multi-class classification
    class_names=CLASS_NAMES,
    validation_split=VALIDATION_SPLIT,
    subset="both",
    seed=123,  # for reproducibility
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale'  # Load as grayscale [cite: 16]
)

# %% [markdown]
# --- Load Data for Transfer Learning (RGB) ---<br>
# Most pre-trained models expect 3 color channels.<br>
# We'll load as RGB; image_dataset_from_directory handles conversion if source is grayscale.

# %%
train_ds_rgb, val_ds_rgb = image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASS_NAMES,
    validation_split=VALIDATION_SPLIT,
    subset="both",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='rgb'  # Load/convert to RGB for pre-trained model
)

# %% [markdown]
# %%<br>
# --- Verify Data Loading ---

# %%
print("Class names (Grayscale):", train_ds_gray.class_names)
print("Class names (RGB):", train_ds_rgb.class_names)

# %%
plt.figure(figsize=(10, 10))
for images, labels in train_ds_gray.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # Squeeze removes the channel dimension for grayscale display
        plt.imshow(np.squeeze(images[i].numpy().astype("uint8")), cmap='gray')
        # Find the index of the '1' in the one-hot encoded label
        label_index = np.argmax(labels[i])
        plt.title(train_ds_gray.class_names[label_index])
        plt.axis("off")
plt.suptitle("Sample Grayscale Images from Training Set")
plt.show()

# %%
plt.figure(figsize=(10, 10))
for images, labels in train_ds_rgb.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_index = np.argmax(labels[i])
        plt.title(train_ds_rgb.class_names[label_index])
        plt.axis("off")
plt.suptitle("Sample RGB Images from Training Set")
plt.show()

# %% [markdown]
# %% [markdown]<br>
# ## 3. Configure Data Performance<br>
# Use buffered prefetching to load images from disk without I/O blocking.

# %% [markdown]
# %%

# %%
AUTOTUNE = tf.data.AUTOTUNE

# %%
train_ds_gray = train_ds_gray.cache().shuffle(
    1000).prefetch(buffer_size=AUTOTUNE)
val_ds_gray = val_ds_gray.cache().prefetch(buffer_size=AUTOTUNE)

# %%
train_ds_rgb = train_ds_rgb.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds_rgb = val_ds_rgb.cache().prefetch(buffer_size=AUTOTUNE)

# %% [markdown]
# %% [markdown]<br>
# ## 4. Model 1: Custom CNN from Scratch

# %% [markdown]
# %%<br>
# --- Define Custom CNN Architecture ---<br>
# Normalization layer adapted to grayscale input range [0, 255]<br>
# Based on Checkpoint 1, pixel values were normalized to [0,1],<br>
# but image_dataset_from_directory loads them as [0, 255].<br>
# We'll rescale here.

# %%
rescale_layer = layers.Rescaling(
    1./255, input_shape=IMG_SHAPE_GRAY)  # [cite: 17]

# %%
model_custom_cnn = Sequential([
    rescale_layer,
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # More dropout before final layer
    layers.Dense(NUM_CLASSES, activation='softmax')  # Softmax for multi-class
])

# %% [markdown]
# %%<br>
# --- Compile Custom CNN ---

# %%
model_custom_cnn.compile(
    optimizer='adam',  # A common starting optimizer
    loss='categorical_crossentropy',  # Standard for multi-class
    metrics=['accuracy']
)

# %% [markdown]
# %%<br>
# --- Print Model Summary ---

# %%
model_custom_cnn.summary()

# %% [markdown]
# %% [markdown]<br>
# ### Train the Custom CNN

# %% [markdown]
# %%

# %%
print("Training Custom CNN...")
history_custom_cnn = model_custom_cnn.fit(
    train_ds_gray,
    validation_data=val_ds_gray,
    epochs=EPOCHS_CUSTOM_CNN
)
print("Custom CNN Training Complete.")

# %% [markdown]
# %% [markdown]<br>
# ## 5. Model 2: Transfer Learning (MobileNetV2)

# %% [markdown]
# %%<br>
# --- Define Transfer Learning Architecture ---

# %% [markdown]
# Preprocessing layer specific to MobileNetV2 (scales input to [-1, 1])

# %%
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale_transfer = layers.Rescaling(1./127.5, offset=-1)  # Scale to [-1, 1]

# %% [markdown]
# Load the base model (pre-trained on ImageNet)<br>
# include_top=False removes the final classification layer

# %%
base_model = MobileNetV2(input_shape=IMG_SHAPE_RGB,
                         include_top=False,
                         weights='imagenet')

# %% [markdown]
# Freeze the base model layers

# %%
base_model.trainable = False

# %% [markdown]
# Build the new model

# %%
inputs = tf.keras.Input(shape=IMG_SHAPE_RGB)
x = preprocess_input(inputs)  # Use the specific preprocessing
# x = rescale_transfer(inputs) # Alternative if preprocess_input causes issues
x = base_model(x, training=False)  # Run base model in inference mode
x = layers.GlobalAveragePooling2D()(x)  # Pool features
x = layers.Dropout(0.3)(x)  # Regularization
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(
    x)  # New classifier head

# %%
model_transfer = tf.keras.Model(inputs, outputs)

# %% [markdown]
# %%<br>
# --- Compile Transfer Learning Model ---<br>
# Use a potentially smaller learning rate for fine-tuning later,<br>
# but Adam often works okay for the initial head training.

# %%
model_transfer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Start with Adam
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %% [markdown]
# %%<br>
# --- Print Model Summary ---

# %%
model_transfer.summary()

# %% [markdown]
# %% [markdown]<br>
# ### Train the Transfer Learning Model (Classifier Head Only First)

# %% [markdown]
# %%

# %%
print("Training Transfer Learning Model (Head Only)...")
history_transfer_head = model_transfer.fit(
    train_ds_rgb,
    validation_data=val_ds_rgb,
    epochs=EPOCHS_TRANSFER  # Usually requires fewer epochs for the head
)
print("Transfer Learning (Head Only) Training Complete.")

# %% [markdown]
# %% [markdown]<br>
# ### Optional: Fine-tuning the Transfer Learning Model<br>
# Uncomment the following cells to unfreeze some top layers of the base model and continue training with a very low learning rate.

# %%
base_model.trainable = True
fine_tune_at = 100 
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False
  
model_transfer.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_transfer.summary()


# %%
fine_tune_epochs = 10
total_epochs = EPOCHS_TRANSFER + fine_tune_epochs

history_fine_tune = model_transfer.fit(
    train_ds_rgb,
    epochs=total_epochs,
    initial_epoch=history_transfer_head.epoch[-1] + 1,
    validation_data=val_ds_rgb
)

# %% [markdown]
# %% [markdown]<br>
# ## 6. Visualize Training Results

# %% [markdown]
# %%<br>
# --- Plotting Function ---

# %%
def plot_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy\n({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title(f'Training and Validation Loss\n({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# %%<br>
# --- Plot Custom CNN History ---

# %%
print("Plotting Custom CNN Training History...")
plot_history(history_custom_cnn, "Custom CNN")

# %% [markdown]
# %%<br>
# --- Plot Transfer Learning History ---<br>
# If you fine-tuned, you'll need to combine histories or plot separately.<br>
# Plotting the initial head training here:

# %%
print("Plotting Transfer Learning (Head) Training History...")
plot_history(history_transfer_head, "Transfer Learning (Head Only)")

# %% [markdown]
# If fine-tuning was performed and you want to plot the combined history:<br>
# combined_history = {}<br>
# for key in history_transfer_head.history.keys():<br>
#    combined_history[key] = history_transfer_head.history[key] + history_fine_tune.history[key]<br>
# plot_history(type('obj', (object,), {'history': combined_history})(), "Transfer Learning (Fine-Tuned)")

# %%
combined_history = {}
for key in history_transfer_head.history.keys():
   combined_history[key] = history_transfer_head.history[key] + history_fine_tune.history[key]
plot_history(type('obj', (object,), {'history': combined_history})(), "Transfer Learning (Fine-Tuned)")

# %% [markdown]
# %% [markdown]<br>
# ## 7. Further Steps (Evaluation)<br>
# - Evaluate the final models on a separate test set (if available).<br>
# - Generate classification reports and confusion matrices.<br>
# - Use the trained models to make predictions on new images.


