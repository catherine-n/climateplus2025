# Modeling : YOLO

이 코드는 Object detection을 위한 YOLO 모델이다. Solar Objects 전체를 하나의 클래스로 인식하는 Single Class, 그리고 3개(Solar Panel, Water heater, Pool heater)를 구분하는 Multi Class 모델로 나뉜다. 
Note : Single Class 모델은 Multi-Class 모델의 label 데이터를 재사용해 모든 Class를 하나의 값(`0`)으로 변경하여 적용했다. 또한, Single Class 코드 내에, False Positive, False Negative, Ground Truth 시각화 코드가 포함돼 있다.

![Image](https://github.com/user-attachments/assets/15d0d163-3d0d-4b2d-b242-96731fbb7614)


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

* Reconstruct a 12,500 × 12,500 image from all U-Net output patches.
* Use `.csv` metadata from YOLO tiles to aligh with its original location
* Resize Ground Truth masks to 320 × 320 for patch-level comparison.
* Use IoU, Precision, and Recall for quantitative evaluation.
* Optionally visualize using visualization_checker.
