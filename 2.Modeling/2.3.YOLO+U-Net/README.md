# Modeling : YOLO + U-Net

This model integrates object detection and semantic segmentation into a unified prediction pipeline.
Both the YOLO (object detector) and U-Net (segmenter) components were independently trained prior to integration.
The combined pipeline allows performance comparison against the standalone U-Net model, using IoU, Precision, and Recall as evaluation metrics.
Furthermore, to assess real-world applicability, the system is designed to estimate the total solar panel area from segmentation outputs.



----
## Step 1: Data Cleaning

* Load the .gpkg files for prediction.
* Integrate the name_qc column (already QC-verified) into the analysis column.
* Merge PV_heater and PV_heater_mat into a unified PV_heater label.
* Save a cleaned .gpkg file.

## Step 2: Data Processing

* Crop input images into 1024 × 1024 tiles.
*Generate corresponding Ground Truth labels for performance evaluation. <br>
(Can be skipped for pure inference-only use cases.)

## Step 3: Prediction

* Run YOLO on each image tile.
* Save predictions in JSON format with confidence threshold ≥ 0.5.
* Optionally save prediction images (frequency adjustable).

> * `image_name`: Name of the image patch
> * `model_type`: YOLO model identifier
> * `num_prediction`: Number of detected solar objectes
> * `prediction_id`: Unique IF for each detected object
> * `class`: Class type - Binary(PV_all) or Multi(Panel, heater, Pool)
> * `confidence`: Confidence score of the model's prediction
> * `polygon`: Pixel coordinated of bounding box

## Step 4: Coordinate converting

* Compute polygon and centroid.
* Convert pixel coordinates to spatial coordinates (CRS: ESRI:102562).

## Step 5: Image Cropping for U-Net

* From YOLO predictions, crop 320 × 320 images centered on each bounding box.
* Cropped patches are passed to U-Net for segmentation.

## Step 6 : U-Net Prediction

* Refer to `Step 3` in U-Net `README` for details.
* Ensure U-Net Ground Truth paths are correctly set for evaluation in `step 7`

## Step 7 : Model evaluaton

* Reconstruct a 12,500 × 12,500 JSON map from all U-Net output patches(Use JSON format, not images)
    - Use Method 2 in the code. (Method 1 uses images, not JSON)
* **Overlapping pixels are resolved by precision priority: Background > Panel > Pool > Heater.**
* Use `.csv` metadata from YOLO tiles to aligh with its original location
* Resize Ground Truth masks to 320 × 320 for patch-level comparison.
* Use IoU, Precision, and Recall for quantitative evaluation.
* Optionally visualize using visualization_checker.
