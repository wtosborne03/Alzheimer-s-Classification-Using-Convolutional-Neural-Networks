# Alzheimer's MRI Classification

This project analyzes MRI scans to classify them into four Alzheimer's stages (AD, CN, EMCI, LMCI) and find early warning signs.

## Files

*   **`eda.ipynb`**: Explores the image data (counts, sizes, pixel values).
*   **`models.ipynb`**: Builds and trains two models:
    1.  A custom CNN (using grayscale images).
    2.  A pre-trained MobileNetV2 (using RGB images).
*   **`custom_model.keras`**: Saved custom CNN model.
*   **`Report.pdf`**: Full project report with details and results.
*   **`*.png`**: Plots showing model training progress.

## Dataset

Uses the "ADNI 4-Class Alzheimer's MRI Classification Dataset" from KaggleHub.

## Process

1.  **Explore Data**: Analyze image properties (`eda.ipynb`).
2.  **Prepare Data**: Load images, resize (190x200), and set up for training (`models.ipynb`).
3.  **Train Models**: Train the custom CNN and the MobileNetV2 model (`models.ipynb`).
4.  **Visualize**: Plot training accuracy and loss (`models.ipynb`).

## Tools Used

*   Python
*   TensorFlow / Keras
*   Pandas, NumPy
*   Matplotlib, Seaborn
*   KaggleHub

## How to Run

1.  Install required libraries (tensorflow, pandas, etc.).
2.  Run `eda.ipynb` for data exploration.
3.  Run `models.ipynb` to train models (needs time/GPU).