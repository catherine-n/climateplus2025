# YOLO

This project implements an object detection model using YOLO (You Only Look Once). The model is trained in two modes:

* **Single-Class**: treats all solar-related objects as a single category.
* **Multi-Clas**s: distinguishes between three separate object types: `Solar Panel`, `Water Heater`, and `Pool Heater`.

**Note**: The Single-Class model reuses the label data from the Multi-Class setup, but converts all class values to a single class label (0).
Additionally, the Single-Class training script includes visualization tools for **False Positives**,** False Negatives**, and** Ground Truths**, to facilitate deeper error analysis.

----
## Step 1: Data Cleaning (Multi-Class Object Detection)

* Load `.gpkg` files to be used for training and inference
* Integrate the name_qc column (already QC-verified) into the columns for analysis
* Merge `PV_heater` and `PV_heater_mat` into a unified PV_heater label.
* Save a cleaned `.gpkg` file.
* The dataset is split into Train/Validation/Test using a 60%/20%/20% ratio with **stratified sampling**
* Stratification is based on treating the full set of object classes per file as a unique unit, ensuring even distribution of class combinations.

**Note**: The lable file names must exactly match the corresponding image file names.

## Step 2: Configuration

* Create a YOLO-compatible `yaml` file that defines:
  * File path for `train`, `val`, and `test` datasets
  * The list of target classes

## Step 3: Training 

* The training script allows configuration of:
  * Model version
  * Image size
  * Number of epochs
  * Batch size
  * Learning rate
  * Other hyperparameters

## Step 4: Evaluaton

* Model evaluation results are saved under the `runs` directory.
* Evaluation is executed as part of the training script.
* Evaluation metrics include: `Precision`, `Recall`, `mAP50`, `mAP50-95`
  
------
## Step 1: Data Cleaning (Single-Class Object Detection)

* Convert the class column (first column) of the Multi-Class label files to a single label value `0`.
* All other preprocessing steps remain the same as in the Multi-Class setup.
  
## Step 2: Configuration

* Adjust the `.yaml` file to reflect binary classification, with only one class defined.

## Step 3,4

* These follow the same procedure as in the Multi-Class setup.

## Step 5: Visualization

* Visual tools are provided to inspect:
  * False Negatives
  * False Positives
  * Ground Truth boxes

![Image](https://github.com/user-attachments/assets/15d0d163-3d0d-4b2d-b242-96731fbb7614)
