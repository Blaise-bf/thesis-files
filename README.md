# Prediction of Central Catheter Tip Position on Chest X-ray images in Renal Disease Patients Subjected to Hemodialysis

---


## Overview

The core goal of this project is to develop a machine learning model that can accurately predict the tip position of central venous catheters (CVCs) in chest X-ray images. This is a critical task for ensuring patient safety during hemodialysis.

## Features

- **Image Segmentation:** Automatically identifies and segments relevant anatomical structures and catheters in X-ray images.
- **Classification:** Predicts the position of the catheter tip (e.g., correct vs. incorrect position).
- Entire workflow is implemented as Jupyter Notebooks for ease of experimentation and reproducibility.

## Requirements

Make sure you have the following packages installed:

- pytorch
- numpy
- torchmetrics
- segmentation_models
- opencv-python
- scikit-learn

You can install the dependencies using pip:

```bash
pip install torch numpy torchmetrics segmentation-models-pytorch opencv-python scikit-learn
```

## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/Blaise-bf/thesis-files.git
    cd thesis-files
    ```
2. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3. Open and run the main notebook (e.g., `thesis_analysis_update.ipynb`) to follow the segmentation and classification workflow.

## Data

- Ensure you have your chest X-ray image dataset prepared and update the notebook paths as needed.

## Results

- Example outputs and performance metrics can be viewed by executing the notebook cells.
- Visualization of segmentation and classification results are included in the notebook.

## License

This project is for academic and research use. Please cite appropriately if you use the code or results.

---

