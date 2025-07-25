# Modeling : YOLO + U-Net

This model integrates object detection and semantic segmentation into a unified prediction pipeline.
Both the YOLO (object detector) and U-Net (segmenter) components were independently trained prior to integration.
The combined pipeline allows performance comparison against the standalone U-Net model, using IoU, Precision, and Recall as evaluation metrics.
Furthermore, to assess real-world applicability, the system is designed to estimate the total solar panel area from segmentation outputs.



----
## Step 1: Image_selection

Prediction 대상이 될 `.gpkg` 파일을 불러온다. (gpkg 폴더를 참조할 것, 총 15개 이미지 

* `CapeTown_ImageIDs.xlxs`: The list of Aerial Images ID and Annotators
* `Aerial Imagery`: Cape Town, 12500 * 12500 size, 8cm/pixel (around 55GB for 2023)

## Step 2: Pre-processing

Load files and clean the data, unify format

* `CRS`: Coordinate Reference System = ESRI:102562
* `Annotator Lists`: Contains a list of annotated images
* `Shapefile`: List of anotations

## Step 3: Bounding Box extraction

* Extract vertices of each aerial image directly from files
* Match annoatations with aerial imagery
* Calculate the area of each polygon using its geometry

## Step 4: Annotation Post-processing

* Drop duplicated annotations correponsing to 'geometry' 
* Merge PV_Pool into PV_pool
* Ensure binary values '1' or 'Nan' for PV related columns
* Create a 'PV_normal' column if all other PV related colums are NaN
* Reorder columns for clarity
* Drop unnecessary columns (layer, path = '2020 layers')
* Reindex 'id' for consistency

## Step 5: Save Dataframe as GPKG

* NOTE: ESRI is not directly readbable in python. Therefore it's safe to use WKT format instead.



* YOLO + U-net is just prediction.
* Folder name will be changed

To-be-developed


README.md

---

Cropped images from `4.Extract_images_centroid.ipynb` are fed into U-Net

To continue the prediction based on YOLO output, please refer to Step 3 in the U-Net README."

---
---
---
